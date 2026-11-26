"""Train on fixed HSSD training houses and choose the displacement on calibration houses."""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import logsumexp
from box_refiner import BoxRefiner, OFFSETS, edge_options, image_channels


def read(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def rgb(path):
    image=cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


def ious(a,b):
    a=np.asarray(a,float).reshape(-1,4)[:,None,:]
    b=np.asarray(b,float).reshape(-1,4)[None,:,:]
    inter=np.maximum(0,np.minimum(a[:,:,2:],b[:,:,2:])-np.maximum(a[:,:,:2],b[:,:,:2])).prod(2)
    union=(a[:,:,2:]-a[:,:,:2]).prod(2)+(b[:,:,2:]-b[:,:,:2]).prod(2)-inter
    return inter/np.maximum(union,1e-6)


def samples(train,preds,data):
    rng=np.random.default_rng(2601)
    feats=[]; losses=[]; weight=[]; examples=defaultdict(int)
    for f in train:
        gt=np.asarray([o["xyxy"] for o in f["boxes"]],float).reshape(-1,4)
        if not len(gt):
            continue
        im=image_channels(rgb(data/f["rgb"]))
        h,w=im.shape[:2]
        def append(box,truth,score,kind,wgt):
            examples[kind]+=1
            for side in range(4):
                pos,features=edge_options(im,box,score,side)
                span=max(2,truth[(side%2)+2]-truth[side%2])
                delta=np.abs(pos-truth[side])/span
                target=np.exp(-(delta-delta.min())/.045)
                feats.append(features); losses.append(target/target.sum()); weight.append(wgt)
        for b in preds[f["frame_id"]]:
            box=np.array(b["xyxy"],float)
            overlaps=ious([box],gt)[0]
            # Weak incidental overlaps cannot establish box-to-instance identity.
            if overlaps.max()>=.20:
                append(box,gt[np.argmax(overlaps)],b["score"],"detector",1.)
        for g in gt:
            size=g[2:]-g[:2]
            for inward in (False,True):
                if inward:
                    sides=np.zeros(4)
                    sides[rng.integers(4)] = rng.uniform(.18,.53)
                    perturbation=sides*np.r_[size,-size]
                else:
                    perturbation=np.tile(size,2)*rng.normal(0,.095,4)
                box=np.clip(g+perturbation,0,[w,h,w,h])
                if np.all(box[2:]-box[:2]>=3):
                    append(box,g,rng.uniform(.35,.8),"jitter_fragment" if inward else "jitter",.25)
    x=np.array(feats,dtype=float); y=np.array(losses); weights=np.asarray(weight)
    return x,y,weights,dict(examples)


def fit(x,y,w):
    mean=x.mean(axis=(0,1)); scale=x.std(axis=(0,1)).clip(.005)
    x=(x-mean)/scale
    def objective(coef):
        logits=x@coef; logs=logits-logsumexp(logits,axis=1)[:,None]
        loss=-(w*(y*logs).sum(1)).sum()/w.sum()+.001*np.sum(coef**2)
        grad=np.einsum("gk,gkd->d",w[:,None]*(np.exp(logs)-y),x)/w.sum()+.002*coef
        return loss,grad
    opt=minimize(objective,np.zeros(x.shape[2]),jac=True,method="L-BFGS-B",options={"maxiter":160})
    if not opt.success:
        raise RuntimeError(opt.message)
    return dict(mean=mean.astype("f"),scale=scale.astype("f"),coef=opt.x.astype("f"))


def update(s,truth,boxes,w,h):
    q=ious([r["xyxy"] for r in boxes],truth)
    i,j=linear_sum_assignment(-((q>=.5)*1000+q))
    good=q[i,j]>=.5; hits=int(good.sum())
    matched=set(j[good])
    small=[n for n,g in enumerate(truth) if (g[2]-g[0])*(g[3]-g[1])<=.02*w*h]
    s["frames"]+=1; s["targets"]+=len(truth); s["boxes"]+=len(boxes); s["hits"]+=hits
    s["small_targets"]+=len(small);s["small_hits"]+=sum(n in matched for n in small)
    s["sum_best_iou"]+=q.max(0).sum() if q.shape[0] else 0.
    s["top1_frames_with_truth"]+=int(bool(truth))
    if len(boxes) and len(truth):
        top=max(range(len(boxes)),key=lambda z:boxes[z]["score"])
        s["top1_hits"]+=int(q[top].max()>=.5)


def final(s):
    div=lambda a,b: float(a/b) if b else 0.
    return dict(frames=int(s["frames"]), targets=int(s["targets"]), proposals=int(s["boxes"]), hits=int(s["hits"]),
                top1_hits=int(s["top1_hits"]), positive_frames=int(s["top1_frames_with_truth"]),
                precision=div(s["hits"],s["boxes"]),recall=div(s["hits"],s["targets"]),
                mean_best_iou_per_target=div(s["sum_best_iou"],s["targets"]),small_targets=int(s["small_targets"]),
                small_hits=int(s["small_hits"]),small_recall=div(s["small_hits"],s["small_targets"]))


def measure(frames,preds):
    scopes=defaultdict(lambda:defaultdict(float))
    by_house=defaultdict(lambda:defaultdict(float))
    for f in frames:
        boxes=preds[f["frame_id"]]
        gt=[b["xyxy"] for b in f["boxes"]]
        for target in (scopes["all"],scopes[f["category"]],by_house[f["house"]]):
            update(target,gt,boxes,384,384)
    return {k:final(v) for k,v in scopes.items()}, {k:final(v) for k,v in by_house.items()}


def save_predictions(path,pred):
    dest=path.with_suffix(".tmp")
    with dest.open("w",encoding="utf-8") as handle:
        for key,rows in pred.items():
            handle.write(json.dumps(dict(frame_id=key,boxes=rows),allow_nan=False)+"\n")
    dest.replace(path)


def run(root,out):
    path=root/"jepa_reid_v21"
    rows=read(path/"data/frames.jsonl"); pred_path=path/"detector/predictions.jsonl"
    pred={f["frame_id"]:f["boxes"] for f in read(pred_path)}
    assert set(pred)=={f["frame_id"] for f in rows}
    splits={k:[f for f in rows if f["split"]==k] for k in ("train","val","test")}
    houses={k:{f["house"] for f in fs} for k,fs in splits.items()}
    assert len(set.union(*houses.values()))==sum(map(len,houses.values()))
    x,y,w,counts=samples(splits["train"],pred,path/"data")
    weights=fit(x,y,w)
    (out/"weights").mkdir(parents=True,exist_ok=True)
    (out/"private").mkdir(exist_ok=True)
    ck=out/"weights/box_refiner.npz"
    np.savez(ck,**weights,strength=np.float32(1.))
    engine=BoxRefiner(ck)
    def apply(frames):
        return {f["frame_id"]:engine.refine(rgb(path/"data"/f["rgb"]),pred[f["frame_id"]]) for f in frames}
    val=splits["val"]
    full=apply(val)
    def blend(scale):
        result={}
        for fid,rows_ in full.items():
            result[fid]=[]
            for source,ref in zip(pred[fid],rows_):
                v=np.array(source["xyxy"]); box=(v+scale*(np.asarray(ref["xyxy"])-v)).tolist()
                result[fid].append(dict(ref,xyxy=box,bbox=box.copy()))
        return result
    calib={s:measure(val,blend(s))[0]["all"] for s in (.0,.25,.5,.75,1.)}
    # Model family must actually move boxes for the experimental comparison.
    # Identity remains a separately evaluated baseline and deployment option.
    rank=lambda s:(calib[s]["hits"],calib[s]["mean_best_iou_per_target"],-s)
    strength=max((.25,.5,.75,1.),key=rank)
    np.savez(ck,**weights,strength=np.float32(strength))
    engine=BoxRefiner(ck)
    new=apply(rows)
    save_predictions(out/"private/refined.jsonl",new)
    t=splits["test"]
    current,base_houses=measure(t,pred); trained,ref_houses=measure(t,new)
    b=np.array([ref_houses[k]["recall"]-r["recall"] for k,r in base_houses.items()])
    effects=defaultdict(lambda:defaultdict(int))
    for f in t:
        gt=[q["xyxy"] for q in f["boxes"]]
        if not gt: continue
        boxes=pred[f["frame_id"]]; moved=new[f["frame_id"]]
        a=ious([r["xyxy"] for r in boxes],gt);z=ious([r["xyxy"] for r in moved],gt)
        for raw,new_iou in zip(a,z):
            slot=raw.argmax(); val0=raw[slot]
            key="fragment" if .05<=val0<.5 else "matched" if val0>=.5 else "off_target"
            s=effects[key]; s["proposals"]+=1;s["same_target_above_half_after"]+=int(new_iou[slot]>=.5)
            s["mean_iou_before"]+=val0;s["mean_iou_after"]+=new_iou[slot]
    for v in effects.values():
        for k in ("mean_iou_before","mean_iou_after"):v[k]/=v["proposals"]
    report=dict(method="RGB boundary alignment on fixed detector proposals",
                split_houses={k:len(v) for k,v in houses.items()},split_frames={k:len(v) for k,v in splits.items()},
                training=counts,training_edges=len(x),strength=strength,calibration=calib,
                calibration_winner_including_identity=max(calib,key=rank),
                checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),paid_calls=0,
                test=dict(dino=current,learned_refiner=trained,paired_proposal_effects=effects,
                          house_macro_recall_delta=float(b.mean()),
                          house_bootstrap_ci95=np.quantile(np.random.default_rng(682).choice(b,(10000,len(b))).mean(1),[.025,.975]).tolist()))
    (out/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(strength=strength,original=current["all"],refined=trained["all"])),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("project",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args(); run(args.project,args.output)
