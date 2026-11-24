"""Fit a weak context prior to natural, physically reached instances absent from the start panorama."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch.nn import functional as F


def goal_features(context, geometry, query):
    """Keep direct target matching separate from other semantic context at the endpoint."""
    n, words = context.shape
    cues = context.copy()
    target = cues[:, query].copy()
    cues[:, query] = 0.
    interaction = np.zeros((n, words, words))
    interaction[:, query, :] = cues
    return np.column_stack((geometry, target, cues, interaction.reshape(n, -1)))


@torch.no_grad()
def records(root, words, anchors, heads, split):
    from report_head import forecast_inputs
    v7, v8 = root / "jepa_separation_v7_20261123", root / "jepa_separation_v8_20261123"
    base = root / "jepa_five_20261123"
    dirs = [v8 / "natural_holdout", v8 / "natural_holdout_aux"] if split == "holdout" else [v7 / "reportable_data", v7 / "aux_reportable"]
    by_start = defaultdict(list)
    for folder in dirs:
        for r in json.loads((folder / (split + ".rows.json")).read_text()):
            by_start[r["sid"]].append(r)
    for sid, rows in by_start.items():
        uid = hashlib.sha256(sid.encode()).hexdigest()
        file = next(f / (uid + ".npz") for f in [base / "clip_v5/views", base / "report_v8/observed_crops"] if (f / (uid + ".npz")).exists())
        with np.load(file) as archive:
            ctx = F.normalize(torch.from_numpy(archive["ctx"].astype("float32")), dim=-1)
        with np.load(Path(rows[0]["raw"]) / "context.npz") as archive:
            plans = archive["planned_actions"].astype(int)
        assert len(ctx) == 12
        headings = torch.arange(12) * (math.pi / 6)
        cache = {}
        for h in (4, 12, 24):
            a = plans[:, :h]
            pred, nearest, _ = forecast_inputs(heads, "full", ctx, headings, torch.tensor(a))
            # Same directional observations and target vocabulary for both learned heads.
            visual = [20. * ((v @ anchors.T).amax(dim=2).mean(dim=1).numpy() - .2) for v in (pred, nearest)]
            yaw = np.cumsum((a == 1).astype(float) - (a == 2), axis=1) * math.pi / 6
            x, y = np.sum((a == 0) * np.cos(yaw), axis=1), np.sum((a == 0) * np.sin(yaw), axis=1)
            geom = np.column_stack(((a == 0).mean(axis=1), np.hypot(x, y)/h, np.sin(yaw[:, -1]), np.cos(yaw[:, -1])))
            cache[h] = (*visual, geom)
        for r in rows:
            h = r["horizon"]
            assert r["action"] == plans[:, :h].tolist()
            pred, obs, geom = cache[h]
            q = words.index(r["query"])
            yield (r["house"], sid, h, q, pred, obs, geom, np.asarray(r["gain_never_any_pixel"]))


def matrix(record, arm, swap=0):
    _, _, _, q, pred, obs, geom, _ = record
    if arm == "geometry":
        return geom
    return goal_features(pred if arm == "forecast" else obs, geom, (q + swap) % pred.shape[1])


def fit(rows, arm):
    pairs, house = [], []
    for row in rows:
        y = row[-1]
        i, j = np.where(y[:, None] > y[None, :])
        if len(i):
            x = matrix(row, arm)
            pairs.append(x[i] - x[j])
            house.extend([row[0]] * len(i))
    d = np.concatenate(pairs)
    _, ids, sizes = np.unique(house, return_inverse=True, return_counts=True)
    scale = (1. / sizes[ids]) / len(sizes)
    def objective(w):
        logits = d @ w
        return np.dot(scale, np.logaddexp(0., -logits)) + .05 * np.dot(w, w), -(scale * expit(-logits)) @ d + .1 * w
    out = minimize(objective, np.zeros(d.shape[1]), jac=True, method="L-BFGS-B")
    if not out.success:
        raise RuntimeError(out.message)
    return out.x, len(d)


def evaluate(rows, heads):
    by_house, by_horizon = defaultdict(list), defaultdict(list)
    total, varying, house_ids, origins, inform_origins = 0, 0, set(), set(), set()
    for row in rows:
        house, sid, h, q, pred, obs, _, y = row
        total += 1
        house_ids.add(house)
        origins.add(sid)
        if len(set(y.tolist())) == 1:
            continue
        varying += 1
        inform_origins.add(sid)
        scores = {name:matrix(row, name) @ w for name, w in heads.items()}
        scores.update(wrong_word=matrix(row, "forecast", 1) @ heads["forecast"], direct_target=pred[:, q], observed_target=obs[:, q])
        results = {name:float(y[np.argmax(s)]) for name,s in scores.items()}
        results.update(uniform=float(np.mean(y)), attainable=float(np.max(y)))
        by_house[house].append(results)
        by_horizon[str(h)].append(results)
    def means(xs):
        return {arm:float(np.mean([r[arm] for r in xs])) for arm in xs[0]} if xs else {}
    house_metrics = [means(xs) for xs in by_house.values()]
    return dict(origins=len(origins), houses=len(house_ids), conditions=total, informative=varying,
                origins_with_informative_choices=len(inform_origins), houses_with_informative_choices=len(by_house),
                house_macro_informative=means(house_metrics),
                per_horizon_informative={h:dict(n=len(xs),metrics=means(xs)) for h,xs in sorted(by_horizon.items())})


def main(root, dest):
    base = root / "jepa_five_20261123"
    sys.path[:0] = [str(base / "report_v7"), str(base / "clip_v5")]
    from route_heads import load_heads
    ck = torch.load(base / "report_v8/v8.pt", weights_only=True, map_location="cpu")
    words, anchors = ck["queries"], ck["grounding"]["anchors"]
    frozen = load_heads(base / "clip_v5/head_v5.pt", base / "sparse_v6/sparse_head.pt")
    torch.set_num_threads(2)
    data = {s:list(records(root, words, anchors, frozen, s)) for s in ("train", "calibration", "holdout")}
    layout = [{r[0] for r in v} for v in data.values()]
    if any(a & b for i,a in enumerate(layout) for b in layout[i+1:]):
        raise ValueError("overlapping houses")
    trained = {arm:fit(data["train"], arm) for arm in ("geometry", "observed", "forecast")}
    weights = {arm:res[0] for arm,res in trained.items()}
    dest.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(dest / "context_weights.npz", **weights)
    result = dict(outcome="count of newly eligible physical instances that had zero pixels in the start history as defined by the natural frontier teacher",
                  protocol="All legal native candidates at 4/12/24 actions. Twelve actually observed panorama headings. House-balanced regularized pairwise logistic fit. Fixed regularization .1; no holdout selection.",
                  pair_samples_train={k:v[1] for k,v in trained.items()}, splits={s:evaluate(rows, weights) for s,rows in data.items()})
    (dest / "natural_hidden.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print("complete", result["splits"]["holdout"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.project_root, args.output)
