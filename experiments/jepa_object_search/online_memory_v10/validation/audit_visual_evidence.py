"""Test the current-image signal that enters spatial memory, on held-out HSSD views."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def metrics(records):
    y = np.asarray([r[0] for r in records], dtype=bool)
    score = np.asarray([r[1] for r in records])
    order = np.argsort(-score, kind="stable")
    ys, ss = y[order], score[order]
    end = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1]
    tp = np.cumsum(ys)[end]
    fp = end + 1 - tp
    auc = ap = None
    if y.any() and not y.all():
        ap = float(np.dot(np.diff(np.r_[0, tp]) / y.sum(), tp / (end + 1)))
        x = np.r_[0, fp] / (~y).sum()
        z = np.r_[0, tp] / y.sum()
        auc = float(np.sum(np.diff(x) * (z[1:] + z[:-1]) / 2))
    return dict(views=len(y), positive=int(y.sum()), prevalence=float(y.mean()), auroc=auc,
                average_precision=ap, median_score_positive=float(np.median(score[y])) if y.any() else None,
                median_score_negative=float(np.median(score[~y])) if (~y).any() else None)


@torch.inference_mode()
def evaluate(index, model_dir):
    sys.path.insert(0, str(model_dir))
    from grounded_route import LocalGrounder

    ck = torch.load(model_dir / "v8.pt", weights_only=True, map_location="cpu")
    net = LocalGrounder(ck["grounding"]["anchors"])
    net.load_state_dict(ck["grounding"])
    net.eval()
    names = list(ck["queries"])
    records = defaultdict(list)
    houses, panels = set(), set()
    for row in json.loads(index.read_text()):
        assert row["split"] == "holdout"
        with np.load(row["observed"]) as data:
            field = next(k for k in ("context_rgb", "ctx_rgb", "rgb", "ctx") if k in data and data[k].ndim == 4)
            digest = hashlib.sha256(np.ascontiguousarray(data[field]).tobytes()).hexdigest()
        if (row["house"], digest) in panels:
            continue
        panels.add((row["house"], digest))
        houses.add(row["house"])
        with np.load(model_dir / "observed_cache" / (digest + ".npz")) as data:
            code = torch.from_numpy(data["code"].astype("float32"))
            sim = torch.from_numpy(data["anchor_sim"].astype("float32"))
        visible, _ = net(code, sim, sim, net.anchors)
        scores = visible.amax(dim=1).numpy()
        with np.load(row["labels"]) as truth:
            for j, name in enumerate(truth["queries"].tolist()):
                if name not in names or not truth["query_present"][j]:
                    continue
                pixels = truth["ctx_grid"][:, j]
                labels = pixels.max(axis=(-1, -2)) >= 20
                assert len(labels) == len(scores)
                records[name].extend(zip(labels.tolist(), scores[:, names.index(name)].tolist()))
    return dict(houses=len(houses), unique_panoramas=len(panels),
                signal="max target visibility across 49 cells of one actually acquired RGB image",
                label="target class has at least 20 semantic pixels in one grid cell",
                unit="view x category, restricted to represented categories in that house",
                overall=metrics([r for rows in records.values() for r in rows]),
                categories={name: metrics(rows) for name, rows in sorted(records.items())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("holdout_index", type=Path)
    parser.add_argument("v8_model_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = evaluate(args.holdout_index, args.v8_model_dir)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result["overall"]))
