"""Coverage of simultaneous matching boxes in single/twin counterfactual rooms."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

from build_frames import read_rows


def iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]))
    h=max(0,min(a[3],b[3])-max(a[1],b[1]))
    intersection=w*h
    return intersection/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def main(root,predictions):
    gt=read_rows(root/"frames.jsonl")
    intended={}
    for r in gt:
        if r["view"]==0 and r["stratum"]=="double":
            assert len(r["boxes"])==2
            intended[r["layout"]]={v["instance"] for v in r["boxes"]}
    guess={r["frame_id"]:r for r in read_rows(predictions)}
    assert set(guess)=={r["frame_id"] for r in gt}
    result={}
    for cat in ("clock","wall_art"):
        for cond in ("single","double"):
            counts=Counter()
            for r in gt:
                if r["category"]!=cat or r["stratum"]!=cond:continue
                labels=r["boxes"];boxes=guess[r["frame_id"]]["boxes"]
                mat=np.array([[iou(l["xyxy"],b["xyxy"]) for b in boxes] for l in labels])
                n=0
                if labels and boxes:
                    a,b=linear_sum_assignment(-((mat>=.5).astype(float)*100+mat))
                    n=int((mat[a,b]>=.5).sum())
                counts["frames"]+=1;counts["label_boxes"]+=len(labels)
                counts["predicted_boxes"]+=len(boxes);counts["matched_boxes"]+=n
                if len(labels)==2:
                    counts["two_targets_visible"]+=1;counts["both_found"]+=n==2
                pair=[b for b in labels if b["instance"] in intended[r["layout"]]]
                if cond=="double" and len(pair)==2:
                    counts["original_and_clone_visible"]+=1
                    if boxes:
                        im=np.array([[iou(l["xyxy"],b["xyxy"]) for b in boxes] for l in pair])
                        ai,bi=linear_sum_assignment(-((im>=.5).astype(float)*100+im))
                        counts["original_and_clone_both_found"]+=int((im[ai,bi]>=.5).sum())==2
                if len(labels)==1:
                    counts["one_target_visible"]+=1
                    counts["one_found"]+=n==1
                    counts["multiple_proposals_for_one_target"]+=len(boxes)>=2
                    counts["two_proposals_both_overlap_same_label"]+=bool(boxes and (mat>=.5).sum()>=2)
            result[f"{cat}_{cond}"]=dict(counts)
    out=root/"controls_detector_audit.json"
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=="__main__":
    main(*map(Path,sys.argv[1:]))
