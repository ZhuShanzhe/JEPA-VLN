"""Paired same-image localization audit. Labels enter only after ranking."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from rank_images import rows,index


def overlaps(pred,gt):
    a=np.asarray(pred,float).reshape(-1,4)
    b=np.asarray(gt,float).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2])
    hi=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.maximum(0,hi-lo).prod(-1)
    return inter/np.maximum(1.e-9,(a[:,2:]-a[:,:2]).prod(-1)[:,None]+(b[:,2:]-b[:,:2]).prod(-1)[None,:]-inter)


def record(pred,gt):
    values=overlaps([r["bbox"] for r in pred],[t["xyxy"] for t in gt])
    counts=dict(frames=1,positive=int(bool(gt)),empty=int(not gt),instances=len(gt),proposals=len(pred))
    for k,key in [(1,"one"),(3,"three"),(len(pred),"all")]:
        sample=values[:k]
        ok=sample>=.50
        if ok.size:
            r,c=linear_sum_assignment(-(ok.astype(float)*(min(ok.shape)+1)+sample))
            n=int(ok[r,c].sum())
        else:
            n=0
        counts[key+"_frames_hit"]=int(n>0)
        counts[key+"_instances_hit"]=n
    counts["top_iou_positive"]=float(values[0].max()) if values.size else 0.
    submitted=bool(pred) and pred[0]["review_margin"]>0
    counts["submitted"]=int(submitted)
    counts["false_empty"]=int(submitted and not gt)
    counts["correct_submitted"]=counts["one_frames_hit"]*int(submitted)
    return counts


def summarize(items):
    if not items:
        return {}
    total={k:sum(r[k] for r in items) for k in items[0]}
    for k in ("one","three","all"):
        total[k+"_positive_frame_rate"]=round(total[k+"_frames_hit"]/total["positive"],4) if total["positive"] else None
        total[k+"_instance_recall"]=round(total[k+"_instances_hit"]/total["instances"],4) if total["instances"] else None
    iou_sum=total.pop("top_iou_positive")
    total["mean_top_iou_positive"]=round(iou_sum/total["positive"],4) if total["positive"] else None
    return total


def finish(project):
    root=project/"jepa_box_v26"
    report={"criterion":"IoU >= 0.50, same image and class; maximum one-to-one matching. All methods use the same frozen CLIP class-relative ranking for top-1 and top-3; all-proposal coverage is an upper bound."}
    for ds in ("hssd","habitat"):
        if ds=="hssd":
            frames=[f for f in rows(project/"jepa_reid_v21/data/frames.jsonl") if f["split"]=="test"]
            gt={f["frame_id"]:f["boxes"] for f in frames}
        else:
            frames=rows(root/"habitat/private/manifest.jsonl")
            gt={r["frame_id"]:[b for b in r["boxes"] if b["pixels"]>=103] for r in rows(root/"habitat/private/gt.jsonl")}
        assert set(index(frames))==set(gt)
        report[ds]={}
        by_arm={}
        digest=set()
        for arm in ("dino","sam","refined"):
            proposals=index(rows(root/"compare/private"/f"{ds}_{arm}.jsonl"))
            assert set(proposals)==set(gt)
            by_group=defaultdict(list)
            house=defaultdict(list)
            for f in frames:
                fid=f["frame_id"]
                rec=record(proposals[fid]["boxes"],gt[fid])
                digest.add(proposals[fid]["model_hash"])
                by_group["total"].append(rec)
                by_group["category_"+f["category"]].append(rec)
                if ds=="habitat":
                    by_group["group_"+f["group"]].append(rec)
                else:
                    house[f["house"]].append(rec)
            report[ds][arm]={k:summarize(v) for k,v in by_group.items()}
            by_arm[arm]={h:v["three_positive_frame_rate"] for h,data in house.items() if (v:=summarize(data))["positive"]}
        assert len(digest)==1
        if ds=="hssd":
            report[ds]["house_paired_three"]={}
            for arm in ("sam","refined"):
                delta=np.array([v-by_arm["dino"][h] for h,v in by_arm[arm].items()])
                rng=np.random.default_rng(2601)
                low,hi=np.quantile(rng.choice(delta,(10000,len(delta))).mean(1),[.025,.975])
                report[ds]["house_paired_three"][arm]=dict(houses=len(delta),minus_dino=float(delta.mean()),ci95=[float(low),float(hi)])
    (root/"compare/box_results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for ds in ("hssd","habitat"):
        for arm in ("dino","sam","refined"):
            print(ds,arm,report[ds][arm]["total"])


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("project",type=Path)
    finish(p.parse_args().project)
