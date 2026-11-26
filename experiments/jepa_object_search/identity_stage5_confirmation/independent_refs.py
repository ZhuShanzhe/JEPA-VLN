"""Frozen reference-selection rule on archived independent HSSD instance masks."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import os
import sys

import cv2
import numpy as np
from PIL import Image


def main(project,out):
    sys.path[:0]=[str(project/"identity_stage5_confirmation"),str(project/"jepa_reid_v21/detector")]
    from trusted_reference import assess_reference,MIN_OCCUPANCY,MIN_DETECTOR_SCORE
    from proposals import LocalCandidateDetector
    import torch
    assert MIN_OCCUPANCY==.3 and MIN_DETECTOR_SCORE==.7
    cv2.setNumThreads(1)
    base=project/"identity_stage3_model/new_houses/private"
    frames=[json.loads(line) for line in (base/"frames.jsonl").read_text().splitlines()]
    boxes={r["name"]:r["boxes"] for r in (json.loads(s) for s in (base/"dino.jsonl").read_text().splitlines())}
    groups=defaultdict(list)
    for row in frames:groups[(row["house"],row["category"])].append(row)
    cohort=[]
    for _,group in sorted(groups.items()):
        cohort.extend(sorted(group,key=lambda r:hashlib.sha256(("independent-ref-v1:"+r["name"]).encode()).hexdigest())[:2])
    model=LocalCandidateDetector(project,device="cpu")
    torch.set_num_threads(2)
    counts=defaultdict(Counter)
    reasons=Counter()
    for r in cohort:
        fid=r["name"];cat=r["category"]
        rgb=np.array(Image.open(base/"observations"/(fid+".png")).convert("RGB"))
        det=model.detect(rgb,cat)
        predictions=[]
        for b in boxes[fid]:
            click=[(b[0]+b[2])/2.,(b[1]+b[3])/2.]
            q=assess_reference(rgb,b,click,det)
            predictions.append((b,click,q))
        # Instance labels are opened only after predicting every proposal.
        with np.load(base/"observations"/(fid+".npz"),allow_pickle=False) as snap:
            mask=snap["semantic"]
        valid={int(i.rsplit(":",1)[1]) for i in r["present"]}
        assert mask.shape==rgb.shape[:2]
        counts[cat]["frames"]+=1
        for b,pt,q in predictions:
            x,y=np.floor(b[:2]).astype(int).clip([0,0],rgb.shape[1::-1])
            u,v=np.ceil(b[2:]).astype(int).clip([0,0],rgb.shape[1::-1])
            px,py=np.floor(pt).astype(int).clip([0,0],np.array(rgb.shape[1::-1])-1)
            sid=int(mask[py,px])
            gt_fill=float(np.mean(mask[y:v,x:u]==sid)) if sid in valid and x<u and y<v else 0.
            good=bool(sid in valid and gt_fill>=.20)
            accepted=q.eligible()
            counts[cat]["candidates"]+=1;counts[cat]["usable"]+=int(good)
            counts[cat]["accepted"]+=int(accepted);counts[cat]["accepted_usable"]+=int(accepted and good)
            reasons["accepted" if accepted else q.reason if q.reason!="supported" else "below_fill"]+=1
    total=sum(counts.values(),Counter())
    stats=lambda c:{**dict(c), "precision":c["accepted_usable"]/c["accepted"] if c["accepted"] else None,
        "recall":c["accepted_usable"]/c["usable"] if c["usable"] else None}
    out.parent.mkdir(exist_ok=True,parents=True)
    result=dict(houses=len({r["house"] for r in cohort}),views=len(cohort),cpu_only=True,
        sampling="First two views by predetermined SHA-256 order per house/category; archived detector boxes and their centres as simulated registration.",
        ground_truth="Habitat instance ID at stored proposal centre belongs to the queried category, and its pixels occupy >=20% of original proposal.",
        detector_floor=MIN_DETECTOR_SCORE,estimated_fill_floor=MIN_OCCUPANCY,
        counts=stats(total),by_category={k:stats(v) for k,v in counts.items()},decisions=dict(reasons),
        cohort_sha256=hashlib.sha256(("\n".join(r["name"] for r in cohort)).encode()).hexdigest())
    with out.open("x") as stream:stream.write(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("output",type=Path)
    main(Path(os.environ["NAV_ROOT"]),ap.parse_args().output)
