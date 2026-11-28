"""Predict future CLIP observations and action-conditioned joint reach/visibility on HSSD."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
SEED = 1128
EPOCHS = 32
VARIANTS = ("joint_long", "short_only", "planar_only", "direct_outcome")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(kind):
    folder=ROOT/("private_"+kind)
    paths=sorted(folder.glob("*.npz"))
    if not paths:
        raise ValueError(f"no {kind} data")
    blocks=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            n=len(data["context"])
            def optional(key,default):
                if key not in data: return np.array(default)
                return np.broadcast_to(data[key], (n,)).copy()
            action=np.asarray(data["action"],np.int64)
            assert action.ndim==2 and action.shape[1]<=128 and np.isin(action,[0,1,2,3]).all()
            assert not np.any((action[:,:-1]==0)&(action[:,1:]!=0)), "actions must be trailing-padded"
            action=np.pad(action,((0,0),(0,128-action.shape[1])))
            future=np.asarray(data["future"],np.float32)
            context=np.asarray(data["context"],np.float32)
            assert context.shape[1] in (1024,2048) and future.shape==(n,1024)
            # First block is always the actual starting image; the optional second is a cue crop.
            context=np.pad(context,((0,0),(0,2048-context.shape[1])))
            pose=np.asarray(data["pose"],np.float32)
            assert np.isfinite(context).all() and np.isfinite(pose).all()
            future_valid=optional("future_valid",np.linalg.norm(future,axis=1)>.5).astype(bool)
            assert np.isfinite(future[future_valid]).all() and np.all(np.linalg.norm(future[future_valid],axis=1)>.1)
            b=dict(context=context,future=np.where(future_valid[:,None],future,0),pose=pose,action=action,
                   future_valid=future_valid,reach=optional("reach",np.full(n,-1)),visible=optional("visible",np.full(n,-1)),
                   house=optional("house",[""]*n).astype(str),group=optional("group",[""]*n).astype(str),
                   split=optional("split",[path.stem.split("-")[0]]*n).astype(str),
                   horizon=(action>0).sum(1),source_file=np.repeat(path.name,n),source_row=np.arange(n),
                   height=optional("height",np.full(n,1.31)),pitch_deg=optional("pitch_deg",np.zeros(n)),
                   native_mask=optional("native_mask",np.ones(n)).astype(bool))
            assert all(b["house"]!="") and all(b["group"]!="")
            assert np.isin(b["split"],["train","val","test"]).all()
            if kind=="pose":
                assert np.isin(b["reach"],[0,1]).all() and np.isin(b["visible"],[-1,0,1]).all()
                assert np.all((b["reach"]==0)|(b["visible"]>=0)), "reachable needs measured visibility"
            blocks.append(b)
    return {k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}


class JointPredictor(nn.Module):
    def __init__(self,dims):
        super().__init__()
        self.scene=nn.Linear(1024,80)
        self.cue=nn.Linear(1024,40,bias=False)
        self.geom=nn.ModuleDict({k:nn.Sequential(nn.Linear(v,40),nn.LayerNorm(40),nn.GELU()) for k,v in dims.items()})
        self.actions=nn.Embedding(4,12,padding_idx=0)
        self.order=nn.GRU(12,28,batch_first=True)
        self.fuse=nn.Sequential(nn.Linear(80+40+40+28+5,128),nn.LayerNorm(128),nn.GELU(),
                                nn.Linear(128,96),nn.GELU())
        self.delta=nn.Linear(96,1024)
        self.readout=nn.Linear(1024,32,bias=False)
        self.outcomes=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,3))

    def forward(self,kind,context,action,pose,variant="joint_long"):
        scene=F.normalize(context[:,:1024],dim=1)
        cue=F.normalize(context[:,1024:],dim=1)
        n=(action!=0).sum(1)
        length=max(1,int(n.max()))
        seq=self.actions(action[:,:length])
        states,_=self.order(seq)
        h=states[torch.arange(len(n),device=n.device),(n-1).clamp(min=0)]
        meta=torch.stack([n/108,(action==1).sum(1)/108,(action==2).sum(1)/108,
                          (action==3).sum(1)/108,torch.full_like(n,kind=="pose")],dim=1)
        sc=self.scene(scene*32); cu=self.cue(cue*32)
        def decode(g):
            hidden=self.fuse(torch.cat((sc,cu,self.geom[kind](g),h,meta),1))
            predicted=F.normalize(scene+self.delta(hidden)*.1,dim=1)
            visual_change=self.readout((predicted-scene)*32)
            if variant=="direct_outcome": visual_change=torch.zeros_like(visual_change)
            return predicted,self.outcomes(torch.cat((hidden,visual_change),1))
        if kind=="pose":
            # Fixed chassis navigation: changing only the optics cannot change body reach.
            # Use the same network with camera-specific inputs hidden for this marginal.
            body=pose.clone()
            body[:,[2,3,6,7,13,14,15,16,17]]=0
            p_body,l_body=decode(body)
            if variant=="planar_only": return p_body,l_body
            predicted,full=decode(pose)
            logits=torch.cat((l_body[:,:1],full[:,1:]),1)
        else:
            predicted,logits=decode(pose)
        return predicted,logits

    def predict(self,kind,context,action,pose,variant="joint_long"):
        future,logits=self(kind,context,action,pose,variant)
        pr,pvr,pvo=logits.sigmoid().unbind(1)
        return future,pr,pvr,pr*pvr,pvo


def normalize_stats(d):
    x=d["pose"][d["split"]=="train"]
    return np.stack([x.mean(0),np.maximum(x.std(0),.05)])


def batch(d,ids,stats,device):
    def t(x,dt): return torch.as_tensor(x,device=device,dtype=dt)
    return (t(d["context"][ids],torch.float32),t(d["action"][ids],torch.long),
            t(np.clip((d["pose"][ids]-stats[0])/stats[1],-8,8),torch.float32))


def loss(model,kind,d,ids,stats,variant,device):
    inputs=batch(d,ids,stats,device)
    pred,logits=model(kind,*inputs,variant)
    valid=torch.as_tensor(d["future_valid"][ids],dtype=torch.bool,device=device)
    terms=[]
    if variant!="direct_outcome" and valid.any():
        actual=F.normalize(torch.as_tensor(d["future"][ids],device=device)[valid],dim=1)
        forecast=pred[valid]
        terms.append((1.-(forecast*actual).sum(1)).mean())
        # Candidate-dependent changes at the same source are sparse in a minibatch;
        # the auxiliary regression below contrasts views only within the same source group.
        groups=d["group"][ids][d["future_valid"][ids]]
        pairs=[]
        for group in np.unique(groups):
            indices=np.flatnonzero(groups==group)
            if len(indices)>1:
                pairs.extend(zip(indices,np.roll(indices,1)))
        if pairs:
            ia,ib=np.asarray(pairs).T
            want=actual[ia]-actual[ib]
            have=forecast[ia]-forecast[ib]
            terms.append(.25*(have-want).square().sum(1).mean())
    if kind=="pose":
        r=torch.as_tensor(d["reach"][ids],device=device,dtype=torch.float32)
        v=torch.as_tensor(d["visible"][ids],device=device,dtype=torch.float32)
        measured=v>=0
        reached=(r==1)&measured
        pr,pvr,pvo=logits.unbind(1)
        terms.append(F.binary_cross_entropy_with_logits(pr,r))
        if reached.any(): terms.append(F.binary_cross_entropy_with_logits(pvr[reached],v[reached]))
        if measured.any(): terms.append(.5*F.binary_cross_entropy_with_logits(pvo[measured],v[measured]))
        terms.append(.5*F.binary_cross_entropy((pr.sigmoid()*pvr.sigmoid()).clamp(1e-6,1-1e-6),r*v.clamp(min=0)))
    return sum(terms) if terms else pred.sum()*0.


def scored(model,kind,d,ids,stats,variant,device):
    with torch.inference_mode():
        outputs=[model.predict(kind,*batch(d,b,stats,device),variant) for b in np.array_split(ids,max(1,int(np.ceil(len(ids)/128))))]
    return [torch.cat(x).cpu().numpy() for x in zip(*outputs)]


def scalar(x): return float(np.mean(x)) if len(x) else None


def cosine(x,y):
    return (x*y).sum(1)/np.maximum(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1),1e-8)


def binary(y,p):
    from sklearn.metrics import roc_auc_score,average_precision_score
    y=np.asarray(y)
    diverse=len(np.unique(y))==2
    return dict(n=len(y),positives=int(y.sum()),brier=scalar((y-p)**2),
                auroc=float(roc_auc_score(y,p)) if diverse else None,
                ap=float(average_precision_score(y,p)) if diverse else None)


def metrics(kind,d,ids,out):
    future,r,v,j,o=out
    actual=d["future"][ids]
    cos=cosine(future,actual)
    copycos=cosine(d["context"][ids,:1024],actual)
    mask=d["future_valid"][ids]
    latent=dict(mean=scalar(cos[mask]),copy=scalar(copycos[mask]),
                by_horizon={str(h):dict(n=int(m.sum()),mean=scalar(cos[m]),copy=scalar(copycos[m]))
                            for h in np.unique(d["horizon"][ids]) if (m:=mask&(d["horizon"][ids]==h)).any()})
    if kind=="long":
        group=np.char.add(np.char.add(d["house"][ids],"/"),d["group"][ids])
        scores= {}
        for h in np.unique(d["horizon"][ids]):
            indices=np.flatnonzero((d["horizon"][ids]==h)&mask)
            hits=[]; chance=[]
            for name in np.unique(group[indices]):
                idx=indices[group[indices]==name]
                if len(idx)<2: continue
                truth=actual[idx]/np.maximum(np.linalg.norm(actual[idx],axis=1,keepdims=True),1e-8)
                # Ignore identical target embeddings: their branch identity cannot be inferred.
                if np.min(1-(truth @ truth.T)+2*np.eye(len(idx)))<1e-6: continue
                sim=future[idx]@truth.T
                best=np.isclose(sim,sim.max(1,keepdims=True),atol=1e-7,rtol=0)
                hits.append(float(np.mean(np.diag(best)/best.sum(1))))
                chance.append(1/len(idx))
            scores[str(h)]=dict(groups=len(hits),top1=scalar(hits),uniform=scalar(chance))
        latent["same_source_branch_retrieval"]=scores
    result=dict(n=len(ids),houses=len(np.unique(d["house"][ids])),latent=latent)
    if kind=="pose":
        y=d["reach"][ids]
        seen=d["visible"][ids]
        joint=(y==1)&(seen==1)
        result.update(reach=binary(y,r),conditional_visible=binary(seen[y==1],v[y==1]),joint=binary(joint,j),
                      optical=binary(seen[seen>=0],o[seen>=0]),
                      joint_by_native={str(val):binary(joint[m],j[m]) for val in (False,True)
                                       if (m:=(d["native_mask"][ids]==val)).any()})
    return result


def run(variant,datasets,stats,seed,device,output):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    dims={k:d["pose"].shape[1] for k,d in datasets.items()}
    model=JointPredictor(dims).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=.01)
    pools={k:np.flatnonzero((d["split"]=="train")&((d["horizon"]<=24) if k=="long" and variant=="short_only" else True)) for k,d in datasets.items()}
    assert all(len(i)>0 for i in pools.values())
    groups={}
    for k,ids in pools.items():
        members={}
        for i in ids:
            d=datasets[k]
            members.setdefault((d["house"][i],d["group"][i]),[]).append(i)
        groups[k]=list(members.values())
    best=float("inf")
    best_weights=None
    for epoch in range(1,EPOCHS+1):
        model.train()
        # Equal source weighting irrespective of numbers of sensor augmentations per house.
        steps=min(100,max(8,int(np.ceil(max(map(len,pools.values()))/96))))
        for _ in range(steps):
            opt.zero_grad()
            total=0
            for k,d in datasets.items():
                if k=="long" and variant=="direct_outcome": continue
                # Sample competing endpoints from a shared source on purpose. Random independent
                # rows would seldom contain sibling futures, especially on the long-path trees.
                bins=groups[k]
                selected=rng.choice(len(bins),size=min(32,len(bins)),replace=False)
                ids=np.concatenate([rng.choice(bins[g],size=min(3,len(bins[g])),replace=False) for g in selected])
                total=total+loss(model,k,d,ids,stats[k],variant,device)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step()
        if epoch%2==0 or epoch==1:
            model.eval()
            ranking=0.
            for k,d in datasets.items():
                ids=np.flatnonzero(d["split"]=="val")
                out=scored(model,k,d,ids,stats[k],variant,device)
                if k=="long":
                    if variant!="direct_outcome":
                        eligible=d["horizon"][ids]<=24 if variant=="short_only" else np.ones(len(ids),bool)
                        ranking+=1-cosine(out[0][eligible],d["future"][ids][eligible]).mean()
                else:
                    y=d["reach"][ids]*(d["visible"][ids]>0)
                    ranking+=((out[3]-y)**2).mean()
            if ranking<best:
                best=ranking
                best_weights=copy.deepcopy(model.state_dict())
                best_epoch=epoch
    model.load_state_dict(best_weights)
    model.eval()
    torch.save(dict(weights={k:v.cpu() for k,v in best_weights.items()},dims=dims,variant=variant,
                    stats={k:torch.tensor(v) for k,v in stats.items()}),output/(variant+".pt"))
    result=dict(epoch=best_epoch,seed=seed,train_rows={k:len(v) if variant!="direct_outcome" or k!="long" else 0 for k,v in pools.items()})
    for k,d in datasets.items():
        ids=np.flatnonzero(d["split"]=="test")
        out=scored(model,k,d,ids,stats[k],variant,device)
        result[k]=metrics(k,d,ids,out)
        if k=="pose" or variant=="joint_long":
            fields=("source_file","source_row","house","group","pose","horizon","native_mask","reach","visible","height","pitch_deg")
            p={field:d[field][ids] for field in fields}
            p.update(index=ids,p_reach=out[1],p_visible_if_reach=out[2],p_joint=out[3],p_optical=out[4])
            if k=="long": p.update(predicted_future=out[0].astype(np.float16),actual_future=d["future"][ids].astype(np.float16))
            filename=f"test_{k}_predictions.npz" if variant=="joint_long" else f"test_{k}_{variant}.npz"
            np.savez_compressed(output/filename,**p)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"private_train")
    parser.add_argument("--variants",nargs="+",choices=VARIANTS,default=VARIANTS)
    args=parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(exist_ok=True)
    data={k:load(k) for k in ("long","pose")}
    houses={s:set(h for d in data.values() for h in d["house"][d["split"]==s]) for s in ("train","val","test")}
    assert all(not houses[a]&houses[b] for a,b in (("train","val"),("train","test"),("val","test")))
    stats={k:normalize_stats(d) for k,d in data.items()}
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(3)
    results={}
    for v in args.variants:
        print("train",v,flush=True)
        results[v]=run(v,data,stats,SEED,device,args.output)
        (args.output/(v+"_metrics.json")).write_text(json.dumps(results[v],allow_nan=False,indent=2)+"\n")
    report=dict(seed=SEED,epochs=EPOCHS,houses={k:len(v) for k,v in houses.items()},variants=results,
                sources={k:{s:sum(d["split"]==s).item() for s in houses} for k,d in data.items()})
    (args.output/"metrics.json").write_text(json.dumps(report,allow_nan=False,indent=2)+"\n")


if __name__=="__main__": main()
