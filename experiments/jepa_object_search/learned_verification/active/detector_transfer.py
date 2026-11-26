"""Evaluate short view decisions on frozen, detector-produced regions in withheld HSSD houses."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from active_view_score import ActiveViewScorer
from train_views import SEED, displacement, rows


def main(project,out):
    prev=project/"jepa_reid_v21"
    sys.path.insert(0,str(prev/"model"))
    from encode_proposals import assignments,ious
    frames=[r for r in rows(prev/"data/frames.jsonl") if r["split"]=="test"]
    assert set(r["house"] for r in frames).isdisjoint(
        r["house"] for r in rows(project/"jepa_reid_v20/data/manifest.jsonl") if r["split"]!="test")
    pred={r["frame_id"]:r for r in rows(prev/"detector/predictions.jsonl")}
    tracks=defaultdict(list)
    for r in frames:
        gt=[b["xyxy"] for b in r["boxes"]]
        det=[b["xyxy"] for b in pred[r["frame_id"]]["boxes"]]
        found={}
        for di,gi in assignments(det,gt).items():
            found[r["boxes"][gi]["instance"]]=(det[di],float(ious([det[di]],[gt[gi]])[0,0]))
        r["found"]=found
        r["camera_xyz"]=np.array(r["T_world_from_camera"])[:3,3]
        tracks[r["sequence"]].append(r)
    models={k:ActiveViewScorer(out/(k+".npz")) for k in ("jepa","direct")}
    cases=[]
    for group in tracks.values():
        for ref in group:
            alt=[r for r in group if r is not ref]
            if len(alt)<2:
                continue
            raw=np.array(Image.open(prev/"data"/ref["rgb"]).convert("RGB"))
            motion=np.asarray([displacement(ref,r) for r in alt])
            for obj,(box,_) in ref["found"].items():
                q=np.array([r["found"].get(obj,(None,0))[1] for r in alt])
                labelled=np.array([any(b["instance"]==obj for b in r["boxes"]) for r in alt])
                possible={k:m.score(raw,box,motion) for k,m in models.items()}
                possible.update(forward_displacement=motion[:,0],smallest_turn=-np.abs(motion[:,2]),
                                random=np.zeros(len(alt)),oracle=q)
                record={}
                for k,rank in possible.items():
                    selected=np.isclose(rank,np.max(rank),atol=1e-6)
                    record[k]=(float(np.mean(q[selected])),float(np.mean(q[selected]>=.5)),
                               float(np.mean(~labelled[selected])),
                               float(np.mean(labelled[selected]&(q[selected]<.5))))
                x0,y0,x1,y1=box
                fraction=(max(x1-x0,0)*max(y1-y0,0))/(raw.shape[0]*raw.shape[1])
                cases.append((ref["house"],ref["category"],bool(np.any(q>=.5) and np.any(q<.5)),fraction,record))
    if not cases:
        raise RuntimeError("No matched reference boxes")
    def measure(c):
        if not c:
            return {"groups":0}
        return dict(groups=len(c),houses=len(set(r[0] for r in c)),metrics={
            model:dict(zip(("mean_iou_of_assigned_detection","prob_same_instance_detected",
                            "prob_no_qualified_gt_region","prob_visible_region_but_detector_missed"),
                np.mean([r[-1][model] for r in c],axis=0).tolist())) for model in c[0][-1]})
    d=defaultdict(list)
    for h,_,_,_,r in cases:
        d[h].append(r["jepa"][1]-r["direct"][1])
    h=np.array([np.mean(v) for v in d.values()])
    result=dict(outcome="true if the fixed detector has a one-to-one IoU>=0.5 assignment to this instance in the chosen endpoint",
        candidates="the two other recorded nearby orientations of the same physical HSSD camera sequence",
        reference_population="only target instances already grounded by a matched live detector box at the reference",
        all=measure(cases),different_detection_outcomes=measure([r for r in cases if r[2]]),
        reference_area_protocol="Area of the matched detector box at the reference divided by full RGB frame area; fixed 0.02 cutoff",
        reference_area_at_most_2pct=measure([r for r in cases if r[3]<=.02]),
        reference_area_over_2pct=measure([r for r in cases if r[3]>.02]),
        small_with_different_detection_outcomes=measure([r for r in cases if r[3]<=.02 and r[2]]),
        by_category={c:measure([r for r in cases if r[1]==c]) for c in sorted(set(r[1] for r in cases))},
        paired_house_macro_jepa_minus_direct_detection=dict(mean=float(np.mean(h)),
             ci95=np.quantile(np.random.default_rng(SEED).choice(h,(5000,len(h))).mean(1),[.025,.975]).tolist()))
    (out/"detector_transfer.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result["all"]),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("project",type=Path)
    parser.add_argument("weights",type=Path)
    a=parser.parse_args()
    main(a.project,a.weights)
