"""Paired held-out house bootstrap of the incremental proposal supplier."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

import numpy as np

from evaluate import counts
from proposals import select


def run(root):
    pred={r["frame_id"]:r for r in map(json.loads,(root/"detector/predictions.jsonl").read_text().splitlines())}
    rows=[r for r in map(json.loads,(root/"data/frames.jsonl").read_text().splitlines()) if r["split"]=="test"]
    grouped=defaultdict(lambda: [Counter(),Counter()])
    for r in rows:
        obs=pred[r["frame_id"]]
        for i,boxes in enumerate((select(obs["raw_boxes"],"base"),obs["boxes"])):
            grouped[r["house"]][i].update(counts(r,boxes))
    rng=np.random.default_rng(210427)
    n=len(grouped)
    draws=rng.multinomial(n,np.repeat(1/n,n),size=5000)
    result={}
    for metric,num,den in (("recall","true_positives","truth"),("two_box_coverage","multiple_with_two_recovered","multiple_frames"),
                           ("unmatched_per_frame","unmatched_proposals","frames")):
        nums=np.array([[p[0][num],p[1][num]] for p in grouped.values()])
        dens=np.array([p[0][den] for p in grouped.values()])
        diffs=(draws@(nums[:,1]-nums[:,0]))/(draws@dens)
        result[metric]={"test_difference":float((nums[:,1]-nums[:,0]).sum()/dens.sum()),
                        "paired_house_bootstrap_95":np.quantile(diffs,[.025,.975]).tolist()}
    out=dict(houses=n,resamples=5000,contrast="frozen calibrated model minus full-frame model",differences=result)
    (root/"detector/house_summary.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out))


if __name__=="__main__":
    run(Path(sys.argv[1]))
