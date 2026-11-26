"""Compare class-agnostic candidate coverage at native output sizes. No semantic selector."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def table(path):
    records=[json.loads(line) for line in path.read_bytes().splitlines()]
    result={r["frame_id"]:r for r in records}
    assert len(result)==len(records)
    return result


def matches(candidates,truth):
    if not len(candidates) or not len(truth):
        return set()
    boxes=np.array([r["xyxy"] for r in candidates],float)[:,None]
    pred_area=np.prod(boxes[...,2:]-boxes[...,:2],axis=-1)
    gt_area=np.prod(truth[:,2:]-truth[:,:2],axis=-1)
    inter=np.maximum(0,np.minimum(boxes[...,2:],truth[:,2:])-np.maximum(boxes[...,:2],truth[:,:2])).prod(-1)
    iou=inter/np.maximum(1.,pred_area+gt_area-inter)
    a,b=linear_sum_assignment(-((iou>=.5)*1000+iou))
    return {int(j) for i,j in zip(a,b) if iou[i,j]>=.5}


def run(project):
    v21=project/"jepa_reid_v21"
    root=project/"jepa_box_v26/sam"
    all_rows=table(v21/"data/frames.jsonl")
    sam=table(root/"private/hssd/rows.jsonl")
    dino=table(v21/"detector/predictions.jsonl")
    subset={k for k,r in all_rows.items() if r["split"]=="test"}
    assert sam.keys()==subset and subset<=dino.keys()
    metrics=defaultdict(Counter)
    house=defaultdict(Counter)
    for k in sorted(subset):
        r=all_rows[k]
        a,b=sam[k]["boxes"],dino[k]["boxes"]
        gt=np.array([o["xyxy"] for o in r["boxes"]],float).reshape(-1,4)
        hit_a,hit_b=matches(a,gt),matches(b,gt)
        small=set(np.flatnonzero(np.prod(gt[:,2:]-gt[:,:2],axis=-1)<=.02*sam[k]["width"]*sam[k]["height"]).tolist())
        counts=Counter(frames=1,positive_frames=int(len(gt)>0),target_appearances=len(gt),
                       native_sam_boxes=len(a),dino_boxes=len(b),
                       native_sam_instances=len(hit_a),dino_instances=len(hit_b),shared_matches=len(hit_a & hit_b),
                       matched_only_by_native_sam=len(hit_a-hit_b),matched_only_by_dino=len(hit_b-hit_a),
                       matched_by_neither=len(gt)-len(hit_a | hit_b),small_appearances=len(small),
                       small_native_sam_matched=len(small & hit_a),small_dino_matched=len(small & hit_b),
                       native_sam_any_frame=int(bool(hit_a)),dino_any_frame=int(bool(hit_b)))
        for group in ("all",r["category"]):
            metrics[group].update(counts)
        house[r["house"]].update(counts)
    keys=sorted(house)
    deltas=np.array([(house[k]["native_sam_instances"]-house[k]["dino_instances"])/house[k]["target_appearances"] for k in keys])
    samples=np.random.default_rng(26).choice(deltas,(5000,len(deltas))).mean(1)
    meta=dict(protocol="Complete frozen HSSD test split; one-to-one IoU >= 0.50 at each generator's delivered pool size. Target appearances, not independent physical instances or rollout scores. Native SAM receives no category; DINO receives the query category. Presence of a matching rectangle is a proposal oracle, not an agent decision.",
              counts=metrics, houses=len(house),delta_sam_dino_house_mean=float(deltas.mean()),
              delta_sam_dino_house_bootstrap95=np.quantile(samples,[.025,.975]).tolist(),
              official_source_sha256=hashlib.sha256((project/"private_sources/MOST-Nav/mapping/sam_backend.py").read_bytes()).hexdigest(),
              manifest_sha256=hashlib.sha256((v21/"data/frames.jsonl").read_bytes()).hexdigest())
    (root/"coverage.json").write_text(json.dumps(meta,indent=2,allow_nan=False)+"\n")
    print(json.dumps(metrics["all"]))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("project",type=Path)
    run(p.parse_args().project)
