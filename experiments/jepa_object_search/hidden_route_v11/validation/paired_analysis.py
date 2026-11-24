"""Paired, house-clustered inference from locked first choices. No fitting."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path

import numpy as np

SEEDS = (11, 29, 47)


def sign_flip(values):
    values = np.asarray(values)
    used = values[np.abs(values) > 1e-12]
    if not len(used):
        return 1.0
    observed = abs(used.sum())
    if len(used) <= 18:
        signs = np.array(list(itertools.product((-1., 1.), repeat=len(used))))
        return float(np.mean(abs(signs @ used) >= observed - 1e-10))
    rng = np.random.default_rng(1147)
    signs = rng.choice((-1., 1.), size=(40000, len(used)))
    return float((1 + np.sum(abs(signs @ used) >= observed-1e-10)) / 40001)


def summarize(rows):
    by_origin = defaultdict(list)
    for house, origin, seed, hit in rows:
        by_origin[house, origin, seed].append(hit)
    by_house = defaultdict(list)
    for (house, _, seed), values in by_origin.items():
        by_house[house, seed].append(np.mean(values, axis=0))
    houses = sorted({k[0] for k in by_house})
    matrix = np.array([[np.mean(by_house[h, seed], axis=0) for seed in SEEDS] for h in houses])
    delta = matrix[:,:,0] - matrix[:,:,1]
    per_house = delta.mean(axis=1)
    n = len(houses)
    bootstrap = np.random.default_rng(711).choice(per_house, (20000, n), replace=True).mean(axis=1)
    leave = (per_house.sum()-per_house)/(n-1) if n > 1 else None
    return dict(houses=n, origins=len(by_origin)//len(SEEDS), origin_queries=len(rows)//len(SEEDS),
                jepa_direct_uniform=matrix.mean(axis=(0,1))[:3].tolist(),
                fixed_candidate_index_rates=matrix.mean(axis=(0,1))[3:].tolist(),
                delta=float(per_house.mean()), ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
                seed_deltas=delta.mean(axis=0).tolist(), house_deltas=dict(zip(houses,per_house.tolist())),
                house_wins_ties_losses=[int(np.sum(per_house > 1e-12)), int(np.sum(abs(per_house)<=1e-12)),
                                        int(np.sum(per_house < -1e-12))],
                delete_one_house_range=None if leave is None else [float(min(leave)), float(max(leave))],
                paired_house_signflip_p_two_sided=sign_flip(per_house))


def holm(table, keys):
    keys = sorted(keys, key=lambda k: table[k]["paired_house_signflip_p_two_sided"])
    last = 0.
    for rank, key in enumerate(keys):
        last = max(last,min(1.,(len(keys)-rank)*table[key]["paired_house_signflip_p_two_sided"]))
        table[key]["holm_p_in_family"] = last


def run(file):
    pairs = defaultdict(dict)
    for line in file.read_text().splitlines():
        v = json.loads(line)
        if v["arm"] not in ("jepa","direct"):
            continue
        assert v["split"] == "test"
        k = (v["source"],v["mode"],v["house"],v["origin"],v["query"],v["seed"])
        if v["arm"] in pairs[k]:
            raise ValueError("duplicate paired decision")
        pairs[k][v["arm"]] = v
    conditions = defaultdict(list)
    for (source,mode,house,origin,word,seed), arms in pairs.items():
        a,b = arms["jepa"], arms["direct"]
        assert a["gain"] == b["gain"] and a["group"] == b["group"] and a["horizon"] == b["horizon"]
        y = np.asarray(a["gain"]) > 0
        assert len(y) == 3
        if not any(y):
            continue
        hit = [float(y[a["first"]]),float(y[b["first"]]),float(np.mean(y)),*y.astype(float)]
        labels = ["all"]
        if source == "hssd":
            labels += ["one_profitable" if y.sum() == 1 else "multiple_profitable", "category:" + word]
        else:
            labels += ["condition:" + str(a["group"])]
        for lab in labels:
            conditions["|".join((source,mode,lab))].append((house,origin,seed,hit))
    table = {k:summarize(v) for k,v in sorted(conditions.items())}
    primary = [k for k in table if k.startswith("hssd|") and k.endswith("|all")]
    holm(table, primary)
    secondary = [k for k,v in table.items() if k not in primary and not k.endswith("|all") and v["houses"] >= 4]
    holm(table,secondary)
    return dict(unit="mean across queries at an origin, origins within house, paired seeds and independent houses",
                seeds=list(SEEDS),primary_family=primary,secondary_family_min_houses=4,
                uncertainty="house percentile bootstrap and paired sign-flip sensitivity; descriptive for small house counts",
                table=table)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("predictions", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    output = run(args.predictions)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False))
