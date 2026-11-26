"""One-decision cross-dataset view-quality diagnostic on the untouched official smoke rows."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image


def iou(box,gt):
    if box is None or gt is None:
        return 0.
    a=np.asarray(box); b=np.asarray(gt)
    intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    union=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection
    return float(intersection/union) if union>0 else 0.


def transform(source,views):
    f=np.array(source["camera_fwd_xz"],dtype=float)
    f=f/np.linalg.norm(f)
    right=np.array([-f[1],f[0]])
    out=[]
    for dest in views:
        delta=(np.array(dest["camera_position"])-np.array(source["camera_position"]))[[0,2]]
        g=np.array(dest["camera_fwd_xz"])
        out.append([delta@f,delta@right,np.arctan2(g@right,g@f)])
    return np.array(out)


def score(root,project,out,device,weak_source_head=False):
    sys.path.insert(0,str(project/"jepa_reid_v25/active"))
    sys.path.insert(0,str(project/"jepa_reid_v21/detector"))
    from active_view_score import ActiveViewScorer
    from proposal_worker import ProposalWorker
    weights=project/"jepa_reid_v25/active/weights"
    scorer=ActiveViewScorer
    if weak_source_head:
        sys.path.insert(0,str(project/"jepa_reid_v25/active_weak"))
        from weak_view_score import WeakViewScorer
        scorer=WeakViewScorer
        weights=project/"jepa_reid_v25/active_weak/weights"
    models={n:scorer(weights/(n+".npz")) for n in ("direct","jepa")}
    worker=ProposalWorker(project/"b_verification/env/bin/python",project,device,mode="base")
    index=root/"val/pv_index_50.jsonl"
    rows=[json.loads(x) for x in index.read_bytes().splitlines()]
    cache=out.parent/"detector_cache"; cache.mkdir(exist_ok=True)
    def rgb(path):
        return np.array(Image.open(path).convert("RGB"))
    def detect(path,query):
        key=hashlib.sha256((str(path.relative_to(root))+"|"+query+"|base").encode()).hexdigest()
        entry=cache/(key+".json")
        if entry.exists():
            detections=json.loads(entry.read_bytes())
        else:
            detections=worker.detect(rgb(path),query)
            entry.write_text(json.dumps(detections,allow_nan=False))
        top=max(detections,key=lambda d:d["score"],default=None)
        return top["bbox"] if top else None
    counts=Counter(); details=[]
    try:
        for row in rows:
            counts["rows"]+=1
            base=(root/row["meta_path"]).parent
            doc=json.loads((root/row["meta_path"]).read_bytes())
            views=sorted((r for r in doc["captures"] if r.get("navigable") and r.get("rgb")),key=lambda r:r["tag"])
            if len(views)<3:
                counts["insufficient_poses"]+=1
                continue
            # A deterministic far view, chosen by topology and row identity alone.
            first=[c for c in views if c["range_label"]=="far"] or views
            seed=int(hashlib.sha256(row["episode_path"].encode()).hexdigest()[:8],16)
            ref=first[seed%len(first)]; alt=[c for c in views if c["tag"]!=ref["tag"]]
            moves=transform(ref,alt)
            cat=row["query_object_category"]
            source_path=base/ref["rgb"]; bbox=detect(source_path,cat)
            if bbox is None:
                counts["no_source_detection"]+=1
                continue
            counts["source_has_detector_box"]+=1
            source_rgb=rgb(source_path)
            scores={name:model.score(source_rgb,bbox,moves) for name,model in models.items()}
            scores["least_travel"]=-np.hypot(moves[:,0],moves[:,1])
            scores["least_rotation"]=-np.abs(moves[:,2])
            scores["forward_displacement"]=moves[:,0]
            selected={name:int(np.argmax(values)) for name,values in scores.items()}
            # Only now read evaluation labels. FP/mismatched initial clues are counted.
            if row["pair_type"]=="neg_diff":
                counts["different_query_class_with_box"]+=1
                continue
            initial_iou=iou(bbox,ref.get("mask_bbox_xyxy"))
            if initial_iou<.5:
                counts["same_query_class_source_iou_below_half"]+=1
            quality=np.array([iou(detect(base/c["rgb"],cat),c.get("mask_bbox_xyxy")) for c in alt])
            serial={name:(float(quality[i]),float(quality[i]>=.5)) for name,i in selected.items()}
            serial["uniform_expected"]=(float(np.mean(quality)),float(np.mean(quality>=.5)))
            serial["best_posthoc"]=(float(np.max(quality)),float(np.max(quality)>=.5))
            serial["same_observation"]=(initial_iou,float(initial_iou>=.5))
            details.append(dict(pair=row["pair_type"],scene=row["scene"],case=row["episode_path"],values=serial,
                                good_source=initial_iou>=.5,informative=float(np.ptp(quality))>.01))
    finally:
        worker.close()
    def summary(sub):
        if not sub:
            return dict(rows=0)
        delta=np.array([e["values"]["jepa"][0]-e["values"]["direct"][0] for e in sub])
        by_house={h:delta[[e["scene"]==h for e in sub]] for h in sorted({e["scene"] for e in sub})}
        house_delta=np.array([v.mean() for v in by_house.values()])
        rng=np.random.default_rng(2711)
        samples=house_delta[rng.integers(len(house_delta),size=(3000,len(house_delta)))].mean(axis=1)
        return dict(rows=len(sub),physical_captures=len({e["case"] for e in sub}),houses=len({e["scene"] for e in sub}),
                    rows_with_variant_endpoint_iou=sum(e["informative"] for e in sub),
                    jepa_vs_direct_wins_ties_losses=[int((delta>1e-8).sum()),int((np.abs(delta)<=1e-8).sum()),int((delta<-1e-8).sum())],
                    house_macro_iou_difference_jepa_minus_direct=round(float(house_delta.mean()),4),
                    descriptive_house_bootstrap_interval=list(np.round(np.percentile(samples,[2.5,97.5]),4)),
                    methods={n:dict(mean_top_box_iou=round(float(np.mean([e["values"][n][0] for e in sub])),4),
                                    localized_rate_at_half=round(float(np.mean([e["values"][n][1] for e in sub])),4))
                             for n in sub[0]["values"]})
    good=[r for r in details if r["good_source"]]
    result=dict(index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),paid_requests=0,rows_by_pair=dict(Counter(r["pair_type"] for r in rows)),
                frozen_weight_sha256={k:hashlib.sha256((weights/(k+".npz")).read_bytes()).hexdigest() for k in models},
                counts=dict(counts),all_eligible=summary(good),positive=summary([r for r in good if r["pair"]=="positive"]),
                same_category_distractor=summary([r for r in good if r["pair"]=="neg_same"]),
                posthoc_weak_or_mistaken_source=summary([r for r in details if not r["good_source"]]))
    if weak_source_head:
        result["scope"]="Development reuse of the same known PInVerify smoke subset for a new weak-source-trained head"
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,allow_nan=False))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("data",type=Path);p.add_argument("project",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--device",default="cuda")
    p.add_argument("--weak-source-head",action="store_true",help="Evaluate the separate weak-source-trained weights")
    args=p.parse_args()
    score(args.data,args.project,args.output,args.device,args.weak_source_head)
