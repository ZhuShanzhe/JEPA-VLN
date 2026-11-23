"""Fit current/after-one-turn evidence using teacher supervision only on training houses."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from v9_core import Local,ReviewHead,descriptors,field_labels

D=Path(__file__).resolve().parent
C=D.parent/"collection"
HIST=(-1,0,1)
SEED=91


def write(file,value):
    with file.open("x") as f:json.dump(value,f,indent=2,allow_nan=False)


def features(row):
    with np.load(D/"features"/(row["sid"]+".npz")) as z:
        return torch.from_numpy(z["code"].astype("float32")),torch.from_numpy(z["sim"].astype("float32"))


def teacher(row,words):
    with np.load(row["labels"]) as z:
        grid=z["grid"];near=z["near_grid"];names=z["class_name"].tolist()
        ids=z["instance_id"]; reportable=field_labels(grid,z["eligible"]);prov=str(z["provenance"])
        exclusions=z["excluded_classes"].tolist()
    classes=np.array([words.index(q) for q in names])
    def per_term(arr):
        return np.stack([arr[:,classes==q].sum(1) for q in range(len(words))],1)
    visible=torch.from_numpy(per_term(grid).reshape(12,len(words),49).transpose(0,2,1).astype("float32"))
    nearby=torch.from_numpy(per_term(near).reshape(12,len(words),49).transpose(0,2,1).astype("float32"))
    mask=np.array([q in names and q not in exclusions and q.split(".")[0] not in exclusions for q in words],bool)
    return dict(visible=visible,nearby=nearby,ready=reportable,ids=ids,classes=classes,valid=mask,provenance=prov)


def y_by_term(t,queries,exclude=None):
    out=np.stack([t["ready"][:,(t["classes"]==q)&(t["ids"]!=(-1 if exclude is None else exclude))].any(1) for q in range(queries)],1)
    return np.stack([out,np.roll(out,-1,0),np.roll(out,1,0)],-1).astype("float32")


def local_fit(xs,words):
    model=Local(len(words))
    optim=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.01)
    generator=np.random.default_rng(SEED)
    for epoch in range(10):
        for ids in np.array_split(generator.permutation(len(xs)),max(1,len(xs)//12)):
            code=torch.cat([xs[i][0] for i in ids])
            sim=torch.cat([xs[i][1] for i in ids])
            t=torch.cat([xs[i][2]["visible"] for i in ids]);n=torch.cat([xs[i][2]["nearby"] for i in ids])
            visible,near=model(code,sim)
            def loss(p,y):
                z=(y/60).clamp(0,1)
                return F.binary_cross_entropy(p.clamp(1e-6,1-1e-6),z,weight=1+8*z)
            value=loss(visible,t)+.6*loss(near,n)
            optim.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3.);optim.step()
    return model.eval()


def history_features(x,h,mem):
    prev=torch.roll(x,-h,0)
    d=torch.full_like(x[:,:,0:1],h)
    return torch.cat([x,prev-x,d,d.abs(),torch.full_like(d,float(mem))],-1)


def make_train(rows,xs,local,words):
    x_all=[];y_all=[]
    memory=defaultdict(dict)
    repeat=dict(memory_query_origins=0,removed_positive_views=0,other_instance_positive_views=0)
    for row,(code,sim,t) in zip(rows,xs):
        with torch.no_grad():
            v,n=local(code,sim); x=descriptors(v,n,sim,code)
            y=y_by_term(t,len(words))
            def add(a,b,valid,mem):
                for h in HIST:
                    x_all.append(history_features(a,h,mem)[:,valid,:].reshape(-1,43))
                    y_all.append(torch.from_numpy(b[:,valid,:].reshape(-1,3).copy()))
            add(x,y,t["valid"],False)
            bank=memory[row["house"]]
            for q in np.flatnonzero(t["valid"]):
                options=[i for i in t["ids"][t["classes"]==q] if i in bank]
                if not options:continue
                # Deterministic history from an earlier distinct physical observation.
                key=hashlib.sha256((row["sid"]+"|"+str(q)).encode()).digest()[0]
                iid=options[key%len(options)]
                qx=descriptors(v[:,:,q:q+1],n[:,:,q:q+1],sim[:,:,q:q+1],code,bank[iid])
                yy=y_by_term(t,len(words),iid)[:,q:q+1]
                repeat["memory_query_origins"]+=1
                repeat["removed_positive_views"]+=int((y[:,q,0]>yy[:,0,0]).sum())
                repeat["other_instance_positive_views"]+=int(yy[:,0,0].sum())
                add(qx,yy,np.array([True]),True)
        with np.load(D/"memory"/(row["sid"]+".npz")) as z:
            for i,c in zip(z["ids"],z["vectors"]):bank[int(i)]=torch.from_numpy(c.copy())
    return torch.cat(x_all),torch.cat(y_all),repeat


def score_panel(net,local,rows,out):
    out.mkdir(exist_ok=False)
    index={}
    with torch.no_grad():
        for row in rows:
            code,sim=features(row);v,n=local(code,sim); x=descriptors(v,n,sim,code)
            p=torch.stack([net.layers(history_features(x,h,False)).sigmoid() for h in HIST],1)
            file=out/(hashlib.sha256(row["sid"].encode()).hexdigest()+".npz")
            np.savez_compressed(file,prob=p.numpy())
            index[row["sid"]]=hashlib.sha256(file.read_bytes()).hexdigest()
    write(out/"sha.json",index)


def trials(probs,ys,valid,cut,policy):
    angles=np.broadcast_to(np.arange(12)[:,None],(12,len(valid))).copy()
    qi=np.broadcast_to(np.arange(len(valid)),angles.shape)
    hist=np.ones_like(angles)
    active=np.broadcast_to(valid,angles.shape).copy()
    fire=np.zeros_like(active);seen=ys[angles,qi,0].astype(bool);initial=seen.copy()
    start=angles.copy();steps=np.zeros_like(angles)
    for turn in range(3):
        pred=probs[angles,hist,qi]
        hit=active & (pred[:,:,0]>=cut)
        fire|=hit;active&=~hit
        if turn==2 or policy=="passive":break
        delta=np.where(pred[:,:,1]>=pred[:,:,2],1,-1)
        if policy=="learned":
            active&=(pred[:,:,1:].max(-1)>=.15)&((pred[:,:,1:].max(-1)-pred[:,:,0])>.03)
        elif policy=="fixed":delta=np.where(start%2==0,1,-1)
        angles=(angles+delta*active)%12
        hist=np.where(active,1-delta,hist)
        steps+=active
        seen|=(active & ys[angles,qi,0].astype(bool))
    valid=np.broadcast_to(valid,angles.shape)
    truth=ys[angles,qi,0].astype(bool)
    window=np.stack([ys[(start+k)%12,qi,0] for k in range(-2,3)]).any(0)
    return np.array([valid.sum(),(fire&truth).sum(),(fire&~truth).sum(),(valid&window).sum(),steps.sum(),
                     (valid&~initial&seen).sum(),(valid&~initial&window).sum(),(valid&initial).sum(),(fire&truth&~initial).sum()],int)


def evaluate(rows,words,out,cuts):
    total=defaultdict(lambda:np.zeros(9,int));houses=defaultdict(set)
    for r in rows:
        t=teacher(r,words);y=y_by_term(t,len(words))
        f=out/(hashlib.sha256(r["sid"].encode()).hexdigest()+".npz")
        with np.load(f) as z:p=z["prob"]
        for cut in cuts:
            for pol in ("learned","passive","fixed"):
                v=trials(p,y,t["valid"],cut,pol)
                total[cut,pol,"all"]+=v;total[cut,pol,t["provenance"]]+=v
                if v[1]>0:houses[cut,pol].add(r["house"])
    return total,houses


def fmt(a):
    n,tp,fp,avail,turns,extra,newpossible,initial,tnew=map(int,a)
    return dict(decisions=n,tp=tp,fp=fp,nearby_opportunities=avail,turns=turns,initially_clear=initial,new_evidence=extra,
                initially_hidden_reachable=newpossible,correct_reviews_after_new_view=tnew,
                precision=tp/(tp+fp) if tp+fp else None,reachable_recall=tp/avail if avail else None)


def run(mode):
    torch.manual_seed(SEED);torch.set_num_threads(4)
    words=json.loads((D/"words.json").read_text())
    rows=lambda k:sorted(json.loads((C/(k+".index.json")).read_bytes()),key=lambda r:r["sid"])
    if mode=="fit":
        tr=rows("train");cal=rows("calibration")
        assert len({r["house"] for r in tr})==12 and len({r["house"] for r in cal})==4
        print("loading",len(tr),len(cal),flush=True)
        states=[(*features(r),teacher(r,words)) for r in tr]
        local=local_fit(states,words);x,y,mem=make_train(tr,states,local,words)
        print("examples",len(y),y.mean(0),mem,flush=True)
        heads=ReviewHead();opt=torch.optim.AdamW(heads.parameters(),lr=.001,weight_decay=.015)
        for epoch in range(10):
            for idx in torch.randperm(len(y)).split(2048):
                pred=heads.layers(x[idx]);loss=F.binary_cross_entropy_with_logits(pred,y[idx],pos_weight=torch.full((3,),5.))
                opt.zero_grad();loss.backward();opt.step()
        del x,y,states
        torch.save(dict(words=words,local=local.state_dict(),head=heads.state_dict()),D/"v9.pt")
        score_panel(heads.eval(),local,cal,D/"cal_predictions")
        cuts=[i/100 for i in range(10,100,2)]
        total,houses=evaluate(cal,words,D/"cal_predictions",cuts)
        feasible=[]
        for c in cuts:
            m=fmt(total[c,"learned","all"])
            if m["tp"]+m["fp"]>=12 and (m["precision"] or 0)>=.65 and len(houses[c,"learned"])>=2:
                feasible.append((m["reachable_recall"],m["precision"],c))
        cut=max(feasible)[2] if feasible else None
        policy=dict(threshold=cut,seed=SEED,train_starts=len(tr),calibration_starts=len(cal),supervised_memory=mem,
                    SHA256=hashlib.sha256((D/"v9.pt").read_bytes()).hexdigest(),feature="224-RN50/7x7",threshold_support_decisions=12)
        write(D/"policy.json",policy);print("frozen",policy,flush=True)
        if cut is not None:write(D/"calibration.json",{f"{m}/{g}":fmt(a) for (c,m,g),a in total.items() if c==cut})
    else:
        ck=torch.load(D/"v9.pt",weights_only=True);local=Local(len(words));heads=ReviewHead()
        local.load_state_dict(ck["local"]);heads.load_state_dict(ck["head"]);hold=rows("holdout")
        fit_houses={r["house"] for k in ("train","calibration") for r in rows(k)}
        assert len({r["house"] for r in hold})==4 and not fit_houses&{r["house"] for r in hold}
        pred=D/"blind_predictions";score_panel(heads.eval(),local.eval(),hold,pred)
        pol=json.loads((D/"policy.json").read_text());cut=pol["threshold"] or 1.1
        total,_=evaluate(hold,words,pred,[cut])
        write(D/"blind_results.json",dict(starts=len(hold),houses=4,policy=pol,blind_prediction_manifest=hashlib.sha256((pred/"sha.json").read_bytes()).hexdigest(),scores={f"{m}/{g}":fmt(a) for (c,m,g),a in total.items()}))

if __name__=="__main__":run(sys.argv[1])
