"""Score proposal availability on nearby instances and across controlled camera changes."""
import argparse
from collections import Counter,defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from proposals import iou
from evaluate import counts


def hit_ids(gt,pred):
    if not gt or not pred:
        return set()
    overlap=np.asarray([[iou(a["xyxy"],b["xyxy"]) for b in pred] for a in gt])
    ii,jj=linear_sum_assignment(-((overlap>=.5)*1000+overlap))
    return {gt[i]["instance"] for i,j in zip(ii,jj) if overlap[i,j]>=.5}


def summarize(root):
    data=root/"data"
    directory=root/"detector"
    pred={r["frame_id"]:r for r in map(json.loads,(directory/"predictions.jsonl").read_text().splitlines())}
    frames=list(map(json.loads,(data/"frames.jsonl").read_text().splitlines()))
    assert set(pred)=={f["frame_id"] for f in frames} and len(frames)==len(pred)
    hits,visible,groups={},{},defaultdict(list)
    close=Counter()
    pre_nms,post_nms=Counter(),Counter()
    distinct_pairs=set()
    for f in frames:
        if f["split"]!="test":
            continue
        k=f["frame_id"]
        hits[k]=hit_ids(f["boxes"],pred[k]["boxes"])
        pre_nms.update(counts(f,pred[k]["raw_boxes"]))
        post_nms.update(counts(f,pred[k]["boxes"]))
        visible[k]={b["instance"] for b in f["boxes"]}
        groups[f["sequence"]].append(k)
        for a,b in combinations(f["boxes"],2):
            distance=np.linalg.norm(np.array(a["world_center"])-b["world_center"])
            if distance>1.25:
                continue
            distinct_pairs.add(tuple(sorted([a["instance"],b["instance"]])))
            both=int({a["instance"],b["instance"]}<=hits[k])
            close["close_pair_opportunities"]+=1
            close["close_pair_both_detected"]+=both
            if a["asset"]==b["asset"]:
                close["close_same_asset_opportunities"]+=1
                close["close_same_asset_both_detected"]+=both
    same=Counter()
    for seq in groups.values():
        for a,b in combinations(seq,2):
            for instance in visible[a]&visible[b]:
                same["visible_at_both"]+=1
                same["detected_at_both"]+=int(instance in hits[a]&hits[b])
                same["detected_at_either"]+=int(instance in hits[a]|hits[b])
    summary=dict(test_frames=len(hits),sequences=len(groups),same_view_close=close,multiview=same,
                 distinct_close_pairs=len(distinct_pairs),before_nms=pre_nms,after_nms=post_nms)
    (directory/"coverage_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    summarize(p.parse_args().root)
