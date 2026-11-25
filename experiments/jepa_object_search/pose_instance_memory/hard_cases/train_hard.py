"""Matched-budget hard-impostor sampling and validation-only threshold selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def enrich(rows,kinds,near,assets,legacy):
    # Keep the original number of updates and per-batch strata; only alter
    # relative sampling probabilities within each negative stratum.
    groups={}
    for n,(a,b,y) in enumerate(rows):
        if not y:
            key="legacy" if legacy else (kinds[n][3],kinds[n][5])
            groups.setdefault(key,[]).append(n)
    copies=[]
    hard_total=0
    for group in groups.values():
        idx=np.array(group)
        same=assets[idx]
        if not legacy:
            same=same & np.array([kinds[n][3]=="simultaneous" for n in idx])
        hard=idx[near[idx] | same]
        hard_total+=len(hard)
        if 0<len(hard)<len(idx):
            # Target >=half the negative sampling pool, capped to avoid letting
            # a single unusual training example dominate its category.
            repeat=min(12,max(0,int(np.ceil((len(idx)-2*len(hard))/len(hard)))))
            copies.extend(np.tile(hard,repeat))
    take=np.r_[np.arange(len(rows)),np.array(copies,dtype=int)].astype(int)
    return rows[take],None if kinds is None else [kinds[i] for i in take],dict(
        negative_pairs=sum(map(len,groups.values())),hard_pool=hard_total,extra_sampling_entries=len(copies))


def near_mask(data,p):
    return (data[3][p[:,0]]-data[3][p[:,1]]).norm(dim=1).cpu().numpy()<=.60


def operating_threshold(y,s,near):
    # Cross-view natural detections from validation houses only.
    # Highest repeat recall while both global and close-impostor FPR <=5%.
    neg=y==0
    assert neg.sum() and (near&neg).sum()
    for t in np.unique(np.r_[s,np.nextafter(s.max(),np.float32(np.inf))]):
        hit=s>=t
        if (hit&neg).sum()<=.05*neg.sum() and (hit&neg&near).sum()<=.05*(near&neg).sum():
            return float(t)


def counts(p,score,t,near,label_rows):
    d={}
    for name,mask in [("all",np.ones(len(p),bool)),("within_60cm",near),
                      ("farther_than_60cm",~near),
                      ("same_asset_different_object",label_rows)]:
        y=p[:,2].astype(bool)
        hit=score.astype(float)>=t
        d[name]=dict(repeats=int((y&mask).sum()),recognized=int((hit&y&mask).sum()),
                     impostors=int((~y&mask).sum()),wrong_merge=int((hit&~y&mask).sum()))
    return d


def main(project):
    root=project/"jepa_reid_v21"
    sys.path.insert(0,str(root/"model"))
    from train_pose import load, fit, sequence_pairs,scores
    from pose_head import PoseHead
    data,sets,kinds,props=load(project)
    from train import choose_threshold
    old=[json.loads(l) for l in (project/"jepa_reid_v20/data/manifest.jsonl").read_text().splitlines()]
    assets=np.array([r["asset"] or "" for r in old+props["records"]])
    same=lambda p:(assets[p[:,0]]==assets[p[:,1]])&(p[:,2]==0)
    base,extra=sets["train"]
    baseplus,_,bstat=enrich(base,None,near_mask(data,base),same(base),True)
    extraplus,plusk,estat=enrich(extra,kinds["train"],near_mask(data,extra),same(extra),False)
    # The fit implementation retains the optimizer, epoch, batch/source/label
    # counts, geometric augmentations and validation selection of the original JEPA.
    out=root/"hard_cases"
    trained,ck=fit("pose_jepa",data,dict(sets,train=(baseplus,extraplus)),dict(kinds,train=plusk),out)
    cp=torch.load(root/"model/controls.pt",map_location="cpu",weights_only=True)
    for path in [root/"data/controls/frames.jsonl",root/"detector/controls/predictions.jsonl"]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==cp["hashes"][path.name]
    cr=cp["records"]
    cd=[cp["features"].cuda(),*[torch.tensor(np.array([r[k] for r in cr]),dtype=torch.float32,device="cuda") for k in ("R","camera","point")]]
    control=sequence_pairs(cr,0)["test"]
    control=np.array([r[:3] for r in control if r[3]=="adjacent"],dtype=int)
    ca=np.array([r["asset"] or "" for r in cr])
    def adj(split):
        p=sets[split][1]
        return p[np.array([r[3]=="adjacent" for r in kinds[split]])]
    val,test=adj("val"),adj("test")
    nval,ntest,ncontrol=near_mask(data,val),near_mask(data,test),near_mask(cd,control)
    result=dict(training=dict(seed=12511,epochs=14,updates_per_epoch=80,batch_size=256,
                              original_crop_pool=bstat,detection_pool=estat,selected_epoch=ck["epoch"]),
                scope="Only uniquely assigned detector pairs enter these scores; unknown proposals remain unscored. Calibration houses only determine both thresholds.", methods={})
    for name in ("pose_direct","pose_jepa","hard_jepa"):
        head=trained
        if name!="hard_jepa":
            saved=torch.load(root/"model"/(name+".pt"),map_location="cpu",weights_only=True)
            head=PoseHead().cuda().eval()
            head.load_state_dict(saved["state"])
        mode=name if name!="hard_jepa" else "pose_jepa"
        vs,ts,cs=scores(head,mode,data,val),scores(head,mode,data,test),scores(head,mode,cd,control)
        tgs={"global_5pct":choose_threshold(val[:,2],vs),"global_and_near_5pct":operating_threshold(val[:,2],vs,nval)}
        result["methods"][name]={key:dict(threshold=t,
             val=counts(val,vs,t,nval,same(val)),test=counts(test,ts,t,ntest,same(test)),
             controls=counts(control,cs,t,ncontrol,(ca[control[:,0]]==ca[control[:,1]])&(control[:,2]==0)))
                                  for key,t in tgs.items()}
        if name=="hard_jepa":
            ck.update(thresholds={"detected":tgs["global_and_near_5pct"]},fit="train-only hard-impostor sampling",normal_entry_enabled=False)
            torch.save(ck,out/"hard_jepa.pt")
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    from export_review import export_review
    export_review(project)
    for name,m in result["methods"].items():
        for k,v in m.items():
            print(name,k,v["test"]["all"],v["test"]["within_60cm"],v["controls"]["all"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("project",type=Path)
    main(p.parse_args().project.resolve())
