"""Train matched local proposal heads on counterfactual routes. Evaluation labels never enter decide()."""
from collections import Counter
from pathlib import Path
import json
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from replan import ProposalSet,RepairScorer,RepairHead,embedding_sketch,repair_features,decide,SEED

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"pose_horizon_20261128"
sys.path.insert(0,str(OLD))
from observation_head import HorizonCameraPredictor


def embed(path,predictor,folder=None):
    cache=(folder if folder is not None else HERE/"private_repair")/path.name
    if cache.exists():return
    with np.load(path,allow_pickle=False) as data:
        ctx=data["context"].astype(np.float32);a=data["action"];pose=data["pose"]
        fields={"visual":[],"p_joint":[],"p_visible_if_reach":[]}
        for i in range(0,len(ctx),128):
            x=ctx[i:i+128]
            out=predictor(x,a[i:i+128],pose[i:i+128])
            fields["visual"].append(embedding_sketch(out["predicted_future"],x[:,:1024]))
            for key in ("p_joint","p_visible_if_reach"):fields[key].append(out[key])
        if not len(ctx):return
        np.savez_compressed(cache,**{k:np.concatenate(v) for k,v in fields.items()})


def groups(split,shards=None,checks=None,embeddings=None):
    shards=shards if shards is not None else HERE/"private_data"
    checks=checks if checks is not None else HERE/"private_checker"
    embeddings=embeddings if embeddings is not None else HERE/"private_repair"
    for path in sorted(shards.glob(split+"_*.npz")):
        with np.load(path,allow_pickle=False) as src:
            if not len(src["group"]):continue
            with np.load(checks/path.name,allow_pickle=False) as check, np.load(embeddings/path.name,allow_pickle=False) as vis:
                grp=src["group"];pid=src["plan_id"];a=src["action"];p=src["path"]
                assert np.array_equal(grp,check["group"]) and np.array_equal(pid,check["plan_id"])
                y=src["reach"].astype(bool);v=src["visible"]==1
                c={k:check[k] for k in ("p_reach","suspect_step","suspect_side","decision")}
                feat={k:vis[k] for k in vis.files}
                threshold=np.broadcast_to(check["threshold"],(len(grp),))
                for name in np.unique(grp):
                    ids=np.flatnonzero(grp==name);ids=ids[np.argsort(pid[ids])]
                    assert np.array_equal(pid[ids],np.arange(7))
                    group=ProposalSet(a[ids],p[ids],c["p_reach"][ids],threshold[ids],c["suspect_step"][ids],
                                      c["suspect_side"][ids],c["decision"][ids].astype(str),
                                      feat["visual"][ids],feat["p_joint"][ids],feat["p_visible_if_reach"][ids])
                    yield path.stem,str(name),group,y[ids],v[ids],ids


def learn(split,hints):
    xx=[];yy=[];prev=[];teachers=[];base=[];last=None
    for file,_,s,reach,_,ids in groups(split):
        if last!=file:
            with np.load(HERE/"private_data"/(file+".npz"),allow_pickle=False) as src:
                actual=embedding_sketch(src["future"].astype(np.float32),src["context"][:,:1024].astype(np.float32))
            last=file
        for j in np.flatnonzero(s.decision!="accept"):
            # Supervise all alternatives, including groups with no physically executable route.
            xx.append(repair_features(s,j,hints))
            yy.append(reach.astype(np.float32))
            teachers.append(actual[ids]);base.append(s.visual)
            prev.append(j)
    if not len(xx):raise ValueError("no checker-rejected routes in "+split)
    return np.stack(xx),np.stack(yy),np.asarray(prev),np.stack(teachers),np.stack(base)


def fit(hints,vision=True):
    train,y,j,teacher,base=learn("train",hints)
    val,yv,jv,teachval,baseval=learn("val",hints)
    if not vision:
        for data in (train,val):data[:,:,1:3]=0;data[:,:,11:35]=0
    mean=train.mean((0,1));scale=np.maximum(train.std((0,1)),.03)
    norm=lambda x:torch.from_numpy(np.clip((x-mean)/scale,-7,7))
    x=norm(train);xv=norm(val);y=torch.from_numpy(y);yv=torch.from_numpy(yv)
    def objective(net,x,y,previous,current,future):
        raw=net(x)
        mask=torch.arange(raw.shape[1])[None,:]!=torch.as_tensor(previous)[:,None]
        outcome=F.binary_cross_entropy_with_logits(raw[mask],y[mask])
        correct=y*mask
        mixed=(correct.sum(1)>0)&(correct.sum(1)<mask.sum(1))
        if mixed.any():
            logits=raw[mixed].masked_fill(~mask[mixed],-100)
            target=correct[mixed]/correct[mixed].sum(1,keepdim=True)
            outcome=outcome-.5*(target*F.log_softmax(logits,1)).sum(1).mean()
        if vision and y.bool().any():
            # A measured future embedding is available only for successfully reached poses.
            emb=torch.as_tensor(current)+net.visual_delta(net.net(x))
            outcome=outcome+F.mse_loss(emb[y.bool()],torch.as_tensor(future)[y.bool()])
        return outcome
    torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
    net=RepairHead(len(mean))
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    tag=("with_hint" if hints else "without_hint") if vision else "geometry_only"
    dest=HERE/"private_repair"/(tag+".pt")
    best=float("inf")
    for epoch in range(1,31):
        net.train()
        for ix in np.array_split(rng.permutation(len(x)),max(1,(len(x)+95)//96)):
            l=objective(net,x[ix],y[ix],j[ix],base[ix],teacher[ix])
            opt.zero_grad();l.backward();opt.step()
        net.eval()
        with torch.inference_mode():loss=float(objective(net,xv,yv,jv,baseval,teachval))
        if loss<best:
            best=loss;chosen=epoch
            torch.save(dict(weights=net.state_dict(),mean=torch.tensor(mean),scale=torch.tensor(scale),hints=hints,vision=vision),dest)
    return dict(training_rejected_requests=len(train),validation_rejected_requests=len(val),epoch=chosen)


def evaluate(scored, start_mode):
    counts={}; by_house={};private=[]
    for house,name,s,reached,visible,_ in groups("test"):
        house_rows=by_house.setdefault(house,{})
        initial=0 if start_mode=="straight" else int(np.argmax(s.old_joint))
        for arm in ("original","gate_only","jepa_order","independent","one_shot","geometry_only","without_hint","mask_at_test","trained"):
            op=decide(s,initial,int(max(s.length)),arm,scored.get(arm))
            i=op["index"]
            joint=reached&visible
            row=Counter(groups=1,initial_reach=int(reached[initial]),oracle_reach=int(reached.any()),
                        oracle_joint=int(joint.any()),initial_blocked_with_detour=int(not reached[initial] and reached.any()),
                        initial_rejected=int(s.decision[initial]!="accept"),requested_observation=int(i is None),
                        changed=int(i is not None and i!=initial),
                        issued=int(i is not None),reached=int(i is not None and reached[i]),
                        joint=int(i is not None and joint[i]),issued_blocked=int(i is not None and not reached[i]),
                        saved_blocked=int(not reached[initial] and i is not None and reached[i]),
                        lost_original_reach=int(reached[initial] and (i is None or not reached[i])),
                        issued_nominal_actions=int(0 if i is None else s.length[i]),
                        checks=len(op["history"]))
            # "missed" does not conflate abstention and failed physical execution.
            counts.setdefault(arm,Counter()).update(row)
            house_rows.setdefault(arm,Counter()).update(row)
            private.append(dict(house=house,group=name,arm=arm,initial=initial,selected=i,history=op["history"]))
    diff={}
    rng=np.random.default_rng(SEED)
    for other in ("original","jepa_order","independent","geometry_only","without_hint","mask_at_test"):
        delta=np.array([(values["trained"]["reached"]-values[other]["reached"])/values[other]["groups"]
                         for values in by_house.values()])
        dist=delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1)
        diff[other]=dict(houses=len(delta),macro_difference=float(delta.mean()),
                         interval95=np.quantile(dist,[.025,.975]).tolist())
    (HERE/"private_repair"/(start_mode+"_choices.json")).write_text(json.dumps(private))
    return dict(arms=counts,reach_house_paired=diff,scene_count=len(by_house))


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--encode-only",action="store_true")
    args=p.parse_args();os.umask(0o077)
    torch.set_num_threads(3)
    (HERE/"private_repair").mkdir(exist_ok=True)
    head=HorizonCameraPredictor.from_checkpoints(OLD/"private_train","cuda" if torch.cuda.is_available() else "cpu")
    for path in sorted((HERE/"private_data").glob("*.npz")):embed(path,head)
    if args.encode_only:return
    info={}
    for enabled,arm in ((True,"trained"),(False,"without_hint")):info[arm]=fit(enabled)
    info["geometry_only"]=fit(True,False)
    models={k:RepairScorer(HERE/"private_repair"/n) for k,n in (("trained","with_hint.pt"),("without_hint","without_hint.pt"),("geometry_only","geometry_only.pt"),("mask_at_test","with_hint.pt"))}
    models["mask_at_test"].hints=False
    report={"fit":info,"three_check_ceiling":True}
    for start in ("straight","old_jepa"):
        report[start]=evaluate(models,start)
    (HERE/"repair_results.json").write_text(json.dumps(report,indent=2)+"\n")
    for k,v in report["straight"]["arms"].items():print(k, v["groups"],v["issued"],v["reached"],v["joint"])

if __name__=="__main__":main()
