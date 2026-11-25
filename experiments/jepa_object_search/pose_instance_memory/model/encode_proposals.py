"""Encode real detector boxes. Instance labels enter only the supervised pairing/audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch


def ious(a,b):
    a,b=np.asarray(a).reshape(-1,4),np.asarray(b).reshape(-1,4)
    wh=(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).clip(0)
    overlap=wh[:,:,0]*wh[:,:,1]
    area_a=np.prod((a[:,2:]-a[:,:2]).clip(0),axis=1)
    area_b=np.prod((b[:,2:]-b[:,:2]).clip(0),axis=1)
    return overlap/(area_a[:,None]+area_b[None,:]-overlap).clip(1e-9)


def assignments(pred,labels):
    overlap=ious(pred,labels)
    if not overlap.size:
        return {}
    # Maximize the number over IoU threshold before optimizing the tie-breaking IoU.
    coeff=min(overlap.shape)+1.
    a,b=linear_sum_assignment(-((overlap>=.5)*coeff+overlap))
    return {int(i):int(j) for i,j in zip(a,b) if overlap[i,j]>=.5}


def evidence_image(rgb,box,mask,preprocess):
    height,width=rgb.shape[:2]
    x,y,u,v=box
    def region(r):
        return (max(0,math.floor(x-(u-x)*r)),max(0,math.floor(y-(v-y)*r)),
                min(width,math.ceil(u+(u-x)*r)),min(height,math.ceil(v+(v-y)*r)))
    x0,y0,x1,y1=region(.15)
    a=rgb[y0:y1,x0:x1]
    x0,y0,x1,y1=region(.8)
    b=mask(rgb[y0:y1,x0:x1],(max(0,math.floor(x)-x0),max(0,math.floor(y)-y0),
                              min(x1,math.ceil(u))-x0,min(y1,math.ceil(v))-y0))
    return preprocess(a),preprocess(b)


def support(depth,box,K,pose):
    h,w=depth.shape
    x,y,u,v=box
    x,y,u,v=max(0,int(x)),max(0,int(y)),min(w,math.ceil(u)),min(h,math.ceil(v))
    sub=depth[y:v,x:u]
    valid=np.isfinite(sub)&(sub>.01)&(sub<100)
    if valid.sum()<9:
        return None
    median=np.median(sub[valid])
    vs,us=np.where(valid & (np.abs(sub-median)<=max(.03,.03*median)))
    if not len(us):
        return None
    k=np.argmin((us-np.median(us))**2+(vs-np.median(vs))**2)
    direction=np.linalg.solve(K,np.array([x+us[k]+.5,y+vs[k]+.5,1.]))
    return pose[:3,3]+pose[:3,:3] @ (direction*sub[vs[k],us[k]])


@torch.inference_mode()
def run(args):
    root=args.project/"jepa_reid_v21"
    sys.path.insert(0,str(args.project/"jepa_reid_v20/model"))
    from reid_model import Encoder,mask_context,preprocess
    fpath=args.frames or root/"data/frames.jsonl"
    frames=[json.loads(s) for s in fpath.read_text().splitlines()]
    pp=args.predictions or root/"detector/predictions.jsonl"
    preds={r["frame_id"]:r for line in pp.read_text().splitlines() if (r:=json.loads(line))}
    assert set(preds)=={r["frame_id"] for r in frames} and len(preds)==len(frames)
    out=root/"model"
    cache=args.cache or out/"proposals.pt"
    checks={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (pp,fpath)}
    if cache.exists():
        assert torch.load(cache,map_location="cpu",weights_only=True)["hashes"]==checks
        return
    model=Encoder(args.project).eval().cuda()
    records,labels,arrays,feats=[],{},[],[]
    def flush():
        if arrays:
            a,b=zip(*arrays)
            feats.append(model(torch.stack(a).cuda(),torch.stack(b).cuda()).cpu())
            arrays.clear()
    for fr in frames:
        fid=fr["frame_id"]
        label_boxes=fr["boxes"]
        det=[d["xyxy"] for d in preds[fid]["boxes"]]
        match=assignments(det,[q["xyxy"] for q in label_boxes])
        labels[fid]=[str(q["instance"]) for q in label_boxes]
        rgb=np.array(Image.open(fpath.parent/fr["rgb"]).convert("RGB"))
        depth=np.load(fpath.parent/fr["depth"])
        K,T=np.array(fr["K"]),np.array(fr["T_world_from_camera"])
        for k,box in enumerate(det):
            label=label_boxes[match[k]] if k in match else {}
            # A controlled alternate proposal for the same detection under the same pose.
            # Keep it separate from natural detector outputs in every metric.
            dx,dy=(box[2]-box[0])*.06,(box[3]-box[1])*.06
            perturbed=np.clip(np.array(box)+np.array([dx,dy,-dx,dy]),[0,0,0,0],
                              [rgb.shape[1],rgb.shape[0],rgb.shape[1],rgb.shape[0]])
            for jitter,xyxy in [(False,box),(True,perturbed)]:
                if jitter and not label:
                    continue
                point=support(depth,xyxy,K,T)
                if point is None or min(xyxy[2]-xyxy[0],xyxy[3]-xyxy[1])<2:
                    continue
                arrays.append(evidence_image(rgb,xyxy,mask_context,preprocess))
                records.append(dict(frame=fid,sequence=fr["sequence"],view=fr["view"],split=fr["split"],
                    house=fr["house"],category=fr["category"],slot=k,jitter=jitter,
                    instance=str(label["instance"]) if label else None,asset=label.get("asset"),
                    R=T[:3,:3].tolist(),camera=T[:3,3].tolist(),point=point.tolist()))
                if len(arrays)>=24:
                    flush()
    flush()
    assert records and sum(len(x) for x in feats)==len(records)
    torch.save(dict(features=torch.cat(feats),records=records,labels=labels,hashes=checks),cache.with_suffix(".tmp"))
    cache.with_suffix(".tmp").replace(cache)
    print("encoded",len(records),"proposals including jitter controls",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("project",type=Path)
    for key in ("frames","predictions","cache"):
        p.add_argument("--"+key,type=Path)
    run(p.parse_args())
