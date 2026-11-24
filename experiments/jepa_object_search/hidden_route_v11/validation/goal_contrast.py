"""Compare two goal queries that require disjoint successful branches at the same junction."""
from collections import defaultdict
import argparse
import itertools
import json
from pathlib import Path
import numpy as np


def run(src):
    groups = defaultdict(list)
    for line in src.read_text().splitlines():
        r = json.loads(line)
        if r["source"] == "hssd" and r["split"] == "test":
            groups[r["mode"],r["arm"],r["house"],r["origin"],r["seed"]].append(r)
    by = defaultdict(dict)
    for (mode,arm,house,origin,seed),rows in groups.items():
        vals = []
        for p,q in itertools.combinations(sorted(rows, key=lambda z: z["query"]),2):
            a,b = np.array(p["gain"])>0, np.array(q["gain"])>0
            if not a.any() or not b.any() or (a & b).any():
                continue
            i,j = p["first"],q["first"]
            correct = (float(a[i])+float(b[j]))/2
            exchanged = (float(a[j])+float(b[i]))/2
            vals.append([correct, exchanged, float(a[i] and b[j]), float(i!=j)])
        if vals:
            by[mode,arm][house,origin,seed] = (len(vals),np.mean(vals,0))
    out = {}
    for (mode,arm), data in by.items():
        houses = sorted({k[0] for k in data})
        means = [np.mean([x[1] for k,x in data.items() if k[0]==h],0) for h in houses]
        score = np.mean(means,0)
        out[mode+"|"+arm] = dict(houses=len(houses),origins=len({(k[0],k[1]) for k in data}),
                                 unique_query_pairs=sum(n for n,_ in data.values())//3,
                                 matched=float(score[0]),exchanged=float(score[1]),
                                 both_goals_correct=float(score[2]),different_first_choices=float(score[3]))
    return dict(unit="average over disjoint goal pairs within a junction, origins and seeds within a house, then houses",
                scope="swaps complete first choices, including goal-dependent visual cues in shared memory; diagnostic only",
                arms=out)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("predictions",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    a.output.write_text(json.dumps(run(a.predictions),indent=2,allow_nan=False))
