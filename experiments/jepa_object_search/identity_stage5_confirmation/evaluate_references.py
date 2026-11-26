"""Calibrate on fixed rendered reference proposals; score RGB-only decisions afterwards."""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from trusted_reference import assess_reference, MIN_OCCUPANCY, MIN_DETECTOR_SCORE


def read(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def totals(rows, threshold):
    accepted=[row for row in rows if row[0] is not None and row[0]>=threshold]
    correct=sum(row[1] for row in accepted)
    positives=sum(row[1] for row in rows)
    n=len(accepted)
    # F0.5 emphasizes a clean memory while penalizing a gate which never accepts.
    f05=1.25*correct/(n+.25*positives) if n+.25*positives else 0.
    return dict(candidates=len(rows),usable=positives,accepted=n,usable_accepted=correct,
                precision=correct/n if n else None,recall=correct/positives if positives else None,f05=f05)


def evaluate(project,output):
    base=project/"jepa_box_v26/habitat/private"
    frames=read(base/"manifest.jsonl")
    lookup=lambda name: {r["frame_id"]:r for r in read(base/name)}
    truth=lookup("gt.jsonl")
    dino=lookup("dino.jsonl")
    sam=lookup("sam.jsonl")
    floors=[.4,.55,.65,.70,.75,.80,.85]
    panel={floor:{"calibration":[],"heldout_views":[],"archived":[]} for floor in floors}
    for frame in frames:
        fid=frame["frame_id"]
        part="archived" if frame["group"]=="archived" else (
            "calibration" if int(fid.rsplit("_",1)[-1])%2==0 else "heldout_views")
        rgb=np.asarray(Image.open(base/frame["rgb"]).convert("RGB"))
        h,w=rgb.shape[:2]
        # SAM proposals have a native centroid. Include local detector-origin
        # proposals separately at their box center, matching the registration protocol.
        candidates=[(r["bbox"],r["centroid"]) for r in sam[fid]["boxes"]]
        detections=dino[fid]["boxes"]
        for r in detections:
            b=np.array(r["bbox"])
            candidates.append((b, (b[:2]+b[2:])/2))
        decisions={}
        for floor in floors:
            rows=[r for r in detections if r["score"]>=floor]
            decisions[floor]=[assess_reference(rgb,b,pt,rows,min_score=floor) for b,pt in candidates]
        # Per-instance masks belong exclusively to the post-inference scorer.
        mask=cv2.resize(np.asarray(Image.open(base/truth[fid]["mask"])),(w,h),
                        interpolation=cv2.INTER_NEAREST)
        for i,(b,pt) in enumerate(candidates):
            x,y=np.clip(np.floor(b[:2]).astype(int),[0,0],[w,h])
            u,v=np.clip(np.ceil(b[2:]).astype(int),[0,0],[w,h])
            a,c=np.clip(np.floor(pt).astype(int),[0,0],[w-1,h-1])
            label=mask[c,a]
            target_fill=float(np.mean(mask[y:v,x:u]==label)) if label and x<u and y<v else 0.
            has_pixels=x<u and y<v and bool(np.any(mask[y:v,x:u]))
            failure=("no_task_pixels_in_box" if not has_pixels else "click_misses_task" if not label
                     else "task_at_click_under_20pct" if target_fill<.2 else "usable")
            for floor in floors:
                decision=decisions[floor][i]
                fill=decision.occupancy if decision.reason=="supported" else None
                panel[floor][part].append((fill,int(label>0 and target_fill>=.20),frame["category"],failure))
    thresholds=[.10,.20,.30,.35,.40,.50,.60]
    trials={str(floor):{str(t):totals(panel[floor]["calibration"],t) for t in thresholds} for floor in floors}
    selected_floor,selected=max(((floor,t) for floor in floors for t in thresholds),
                                key=lambda ft:(trials[str(ft[0])][str(ft[1])]["f05"],ft[1],ft[0]))
    report=dict(unit="native SAM region or DINO proposal; pre-commit candidates may be correlated",
                train_and_holdout="even/odd target-facing viewpoints in the same two clock/art development rooms; archived views separate",
                truth="after inference only: click on target instance and same instance covers >=20% of registered ROI",
                objective="best F0.5 on even views, stricter occupancy then detector break ties",thresholds=trials,
                selected_occupancy=selected,selected_detector=selected_floor,
                runtime_occupancy=MIN_OCCUPANCY,runtime_detector=MIN_DETECTOR_SCORE,
                at_selected={part:totals(rows,selected) for part,rows in panel[selected_floor].items()},
                by_category={part:{category:totals([r for r in rows if r[2]==category],selected)
                             for category in sorted({r[2] for r in rows})} for part,rows in panel[selected_floor].items()},
                offered_groups={part:dict(Counter(r[3] for r in rows if r[0] is not None and r[0]>=selected))
                                for part,rows in panel[selected_floor].items()},
                views={group:sum((r['group']=='archived' if group=='archived' else
                    r['group']!='archived' and (int(r['frame_id'].rsplit('_',1)[-1])%2==int(group=='heldout_views')))
                    for r in frames) for group in panel[selected_floor]})
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="thresholds"},allow_nan=False))


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("project",type=Path)
    cli.add_argument("output",type=Path)
    args=cli.parse_args()
    evaluate(args.project,args.output)
