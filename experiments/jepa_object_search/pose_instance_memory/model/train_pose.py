"""Fit matched identity heads on frozen house splits and evaluate natural detector pairs."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

from pose_head import PoseHead,compare


SEED=12511


def legacy_rotations(rows):
    yaw,pitch=np.deg2rad([[r["camera_heading_deg"],r["pitch_deg"]] for r in rows]).T
    forward=np.stack([np.cos(yaw)*np.cos(pitch),np.sin(pitch),np.sin(yaw)*np.cos(pitch)],axis=1)
    right=np.stack([-np.sin(yaw),np.zeros_like(yaw),np.cos(yaw)],axis=1)
    down=np.cross(forward,right)
    return np.stack([right,down,forward],axis=2)


def sequence_pairs(records,offset):
    index=defaultdict(list)
    for i,r in enumerate(records):
        if r["instance"] is not None:
            index[r["sequence"]].append(i)
    result=defaultdict(list)
    for group in index.values():
        for i,j in itertools.combinations(group,2):
            a,b=records[i],records[j]
            if a["jitter"] or b["jitter"]:
                if a["frame"]!=b["frame"] or a["slot"]!=b["slot"] or a["jitter"]==b["jitter"]:
                    continue
                kind="same_view_jitter"
            elif a["frame"]==b["frame"]:
                kind="simultaneous"
            elif abs(a["view"]-b["view"])==1:
                kind="adjacent"
            else:
                continue
            result[a["split"]].append((i+offset,j+offset,int(a["instance"]==b["instance"]),
                                      kind,a["house"],a["category"]))
    return result


def load(project):
    old,root=project/"jepa_reid_v20",project/"jepa_reid_v21"
    sys.path.insert(0,str(old/"model"))
    import train
    manifest=old/"data/manifest.jsonl"
    rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    props=torch.load(root/"model/proposals.pt",map_location="cpu",weights_only=True)
    old_features=torch.load(old/"model/features.pt",map_location="cpu",weights_only=True)
    z=np.load(root/"model/geometry_all.npz",allow_pickle=False)
    assert old_features["manifest_sha256"]==z["manifest_sha256"].item()==hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert np.array_equal(z["ix"],np.arange(len(rows))) and z["valid"].all()
    rec=props["records"]
    feats=torch.cat([old_features["features"],props["features"]]).cuda()
    R=np.concatenate([legacy_rotations(rows),np.stack([r["R"] for r in rec])])
    cameras=np.concatenate([np.array([r["camera_xyz"] for r in rows]),np.stack([r["camera"] for r in rec])])
    points=np.concatenate([z["xyz"],np.stack([r["point"] for r in rec])])
    dataset=[feats,*[torch.tensor(x,dtype=torch.float32,device="cuda") for x in (R,cameras,points)]]
    pair=sequence_pairs(rec,len(rows))
    sets={}
    groups={}
    for split in ("train","val","test"):
        oldpairs=train.make_pairs(rows,split)
        labels=pair[split]
        sets[split]=(oldpairs,np.asarray([r[:3] for r in labels],dtype=int))
        groups[split]=labels
    houses={s:{r["house"] for r in rec+rows if r["split"]==s} for s in sets}
    assert sum(map(len,houses.values()))==len(set.union(*houses.values()))
    return dataset,sets,groups,props


def score_batch(model,mode,data,p,training=False):
    f,R,C,P=data
    i,j=p[:,:2].T
    ca,cb,pa,pb=C[i],C[j],P[i],P[j]
    if training:
        # Common translation error moves the camera and its measurement together
        # within a view; range noise shifts the surface only. Co-frame pairs share drift.
        same_camera=((ca-cb).abs().sum(1,keepdims=True)<1e-5)&((R[i]-R[j]).abs().sum((1,2))[:,None]<1e-5)
        factor=(torch.rand((len(p),1),device=f.device)<.5)
        d1=torch.randn_like(ca)*.15*factor
        d2=torch.where(same_camera,d1,torch.randn_like(cb)*.15*factor)
        ca,cb=ca+d1,cb+d2
        pa,pb=pa+d1+torch.randn_like(pa)*.04*factor,pb+d2+torch.randn_like(pb)*.04*factor
    if mode=="location":
        return -(pa-pb).norm(dim=1),None
    if mode=="pose_forecast":
        from pose_head import conditions
        ab,g=conditions(R[i],ca,pa,R[j],cb,pb)
        ba,_=conditions(R[j],cb,pb,R[i],ca,pa)
        return model(f[i],f[j],ab,ba,torch.zeros_like(g))
    return compare(model,f[i],f[j],R[i],ca,pa,R[j],cb,pb,mode=="heading_jepa")


@torch.no_grad()
def scores(model,mode,data,p):
    parts=[]
    for start in range(0,len(p),512):
        raw,_=score_batch(model,mode,data,p[start:start+512])
        parts.append((raw if mode=="location" else raw.sigmoid()).cpu().numpy())
    return np.concatenate(parts)


def fit(mode,data,sets,kinds,out):
    torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED)
    model=PoseHead().cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.00065,weight_decay=.01)
    base,new=sets["train"]
    # Equal-weight batches for the original orbit crops and the new detector scenes.
    # For the new source, include both co-visible distinct instances and duplicate-box positives.
    slices=[[base[base[:,2]==b]] for b in (0,1)]
    types=np.array([r[3] for r in kinds["train"]])
    cats=np.array([r[5] for r in kinds["train"]])
    for typ,y in (("adjacent",1),("adjacent",0),("simultaneous",0),("same_view_jitter",1)):
        mask=(types==typ)&(new[:,2]==y)
        slices.append([new[mask&(cats==c)] for c in sorted(set(cats[mask]))])
    assert all(len(s) for s in slices)
    vp=sets["val"][1]
    natural=np.array([r[3]=="adjacent" for r in kinds["val"]])
    assert natural.any() and len(np.unique(vp[natural,2]))==2
    best=-1
    for epoch in range(14):
        model.train()
        for _ in range(80):
            pieces=[]
            for strata,n in zip(slices,(64,64,32,32,32,32)):
                counts=rng.multinomial(n,np.ones(len(strata))/len(strata))
                pieces.extend(s[rng.integers(0,len(s),size=k)] for s,k in zip(strata,counts) if k)
            p=np.concatenate(pieces)
            logit,sim=score_batch(model,mode,data,p,training=True)
            y=torch.tensor(p[:,2],dtype=torch.float32,device="cuda")
            loss=F.binary_cross_entropy_with_logits(logit,y)
            if mode!="pose_direct":
                loss=loss+3*(1-sim[y.bool()].mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
            opt.step()
        model.eval()
        va=scores(model,mode,data,vp[natural])
        old_val=sets["val"][0]
        vb=scores(model,mode,data,old_val)
        quality=(roc_auc_score(vp[natural,2],va,max_fpr=.05)+roc_auc_score(old_val[:,2],vb,max_fpr=.05))/2
        if quality>best:
            best=quality
            ck=dict(mode=mode,epoch=epoch+1,validation_pauc=float(best),
                    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    model.load_state_dict(ck["state"])
    return model,ck


def main(args):
    root=args.project/"jepa_reid_v21"
    out=root/"model"
    data,sets,kinds,props=load(args.project)
    from train import choose_threshold,describe
    result={"pair_counts":{s:{"legacy":len(a),"new":len(b)} for s,(a,b) in sets.items()},
            "seed":SEED,"methods":{}}
    pose_questions={}
    for mode in ("location","heading_jepa","pose_forecast","pose_direct","pose_jepa"):
        head,ck=(None,None) if mode=="location" else fit(mode,data,sets,kinds,out)
        stats,ts,track={}, {},{}
        for cohort,index in (("legacy",0),("detected",1)):
            v,test=sets["val"][index],sets["test"][index]
            val=scores(head,mode,data,v)
            sc=scores(head,mode,data,test)
            vmask=np.ones(len(v),dtype=bool) if index==0 else np.array([r[3]=="adjacent" for r in kinds["val"]])
            threshold=choose_threshold(v[vmask,2],val[vmask])
            ts[cohort]=threshold
            if index==0:
                masks={"all":np.ones(len(test),dtype=bool)}
            else:
                typ=np.array([r[3] for r in kinds["test"]])
                cat=np.array([r[5] for r in kinds["test"]])
                near=(data[3][test[:,0]]-data[3][test[:,1]]).norm(dim=1).cpu().numpy()<=.60
                masks={k:typ==k for k in ("adjacent","simultaneous","same_view_jitter")}
                masks["adjacent_estimated_within_60cm"]=(typ=="adjacent")&near
                masks["adjacent_excluding_bookcase"]=(typ=="adjacent")&(cat!="bookcase")
                for c in sorted(set(cat)):
                    masks["adjacent_"+c]=(typ=="adjacent")&(cat==c)
            stats[cohort]={k:describe(test[m,2],sc[m],threshold) for k,m in masks.items() if m.any()}
            if index==1:
                track["test_scores"]=sc
                track["val_scores"]=val
                by_house=defaultdict(list)
                for k,r in enumerate(kinds["test"]):
                    if r[3]=="adjacent" and r[2]:
                        by_house[r[4]].append(int(sc[k]>=threshold))
                pose_questions[mode]=by_house
        np.savez_compressed(out/(mode+"_private_scores.npz"),**track)
        if ck:
            ck["thresholds"]=ts
            torch.save(ck,out/(mode+".pt"))
        result["methods"][mode]={"epoch":ck["epoch"] if ck else None,"thresholds":ts,"metrics":stats}
        print(mode,stats["detected"].get("adjacent"),flush=True)
    keys=sorted(pose_questions["pose_jepa"])
    rng=np.random.default_rng(SEED)
    resample=rng.integers(len(keys),size=(5000,len(keys)))
    result["test_houses_with_matched_revisits"]=len(keys)
    diffs={}
    for method in ("location","heading_jepa","pose_forecast","pose_direct"):
        delta=np.array([sum(pose_questions["pose_jepa"][h])-sum(pose_questions[method][h]) for h in keys])
        den=np.array([len(pose_questions[method][h]) for h in keys])
        distribution=delta[resample].sum(1)/den[resample].sum(1)
        diffs[method]=[float(delta.sum()/den.sum()),*np.quantile(distribution,[.025,.975]).tolist()]
    result["paired_pose_jepa_minus_other_recall_and_house_ci95"]=diffs
    (out/"pose_results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("project",type=Path)
    main(p.parse_args())
