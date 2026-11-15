"""Measure hidden-outcome ambiguity at identical decision-time inputs on Blender bays."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from data_prep import read, records, require, sha, within


def summarize(raw):
    audit=read(raw/"capture_audit.json")
    require(sha(raw/"index.jsonl")==audit["index_sha256"],"captured index changed")
    annotations={row["id"]:row for row in audit["rows"]}
    grouped=defaultdict(lambda:defaultdict(list))
    counts=defaultdict(Counter)
    physical=defaultdict(set)
    for row in records(raw/"index.jsonl"):
        split=row["split"]
        a=annotations[row["id"]]
        with np.load(within(raw,row["path"])) as f:
            h=hashlib.sha256()
            for name in ("context_rgb","xy","actions","step_m","turn_deg"):
                h.update(np.ascontiguousarray(f[name]).tobytes())
            top=np.flatnonzero(f["gain"]==f["gain"].max()).tolist()
            grouped[split][row["house"],row["query"],h.hexdigest()].append(top)
            for i,clip in enumerate(f["future_rgb"]):
                physical[split].add(hashlib.sha256(clip.tobytes()).digest())
                live=(a["first_pixels"][i]<audit["threshold_pixels"] and
                      a["sustained_peak"][i]>=audit["threshold_pixels"])
                require(bool(f["gain"][i])==live,"inconsistent exposure label")
            counts[split]["queries"]+=1
            counts[split]["positive_start_pixels"]+=int(max(a["first_pixels"])>0)
            counts[split]["start_at_exposure_threshold"]+=int(max(a["first_pixels"])>=audit["threshold_pixels"])
            counts[split]["unequal_realized_gain"]+=int(np.ptp(f["gain"])>0)
    result={}
    for split,g in grouped.items():
        conflict=[rows for rows in g.values() if not set.intersection(*(set(s) for s in rows))]
        result[split]=dict(**counts[split],distinct_future_clips=len(physical[split]),visible_query_groups=len(g),
                           contradictory_groups=len(conflict),queries_in_contradictory_groups=sum(map(len,conflict)))
    return dict(index_sha256=audit["index_sha256"],split=result,
                meaning="Equal current RGB, goal, nominal controls, coordinates and geometry, with opposite optimal choices after hidden target swaps. Retain for uncertainty tests. Realized unequal rewards alone do not demonstrate that correct choices can be predicted.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    text=json.dumps(summarize(a.raw),indent=2)+"\n"
    if a.output:
        with a.output.open("x") as f:
            f.write(text)
    print(text,end="")
