"""Fit and blind-test a conservative switch between two already-frozen route experts."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


VIEWS = (12, 1, 2, 3)
RIDGE_PENALTY = 1.
MIN_EXPECTED_GAIN = .025


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def records(index, predictions):
    rows = {r["sid"]: r for r in json.loads(index.read_bytes())}
    frozen = json.loads((predictions / "predictions.json").read_bytes())
    assert len(frozen) == len(rows) and {r["sid"] for r in frozen} == set(rows)
    for record in frozen:
        row = rows[record["sid"]]
        file = predictions / record["file"]
        assert sha(file) == record["sha256"]
        with np.load(file) as p, np.load(row["labels"]) as labels:
            vocab = p["words"].tolist()
            target_words = labels["queries"].tolist()
            cols = [i for i, valid in enumerate(labels["query_present"]) if valid]
            word_idx = [vocab.index(target_words[c]) for c in cols]
            y = labels["gain"][:, :, cols].transpose(1, 0, 2)
            y = np.repeat(y, len(VIEWS), axis=0)
            o, j = p["observed"][:, :, word_idx], p["joint"][:, :, word_idx]
            x = p["x"][:, :, word_idx]
            yo = np.take_along_axis(y, o, axis=1)
            yj = np.take_along_axis(y, j, axis=1)
            yield row, np.array(cols), x, o, j, yo, yj, y


def fit(index, predictions, output):
    groups = list(records(index, predictions))
    assert len(groups) == 16 and len({r["house"] for r, *_ in groups}) == 4
    data, response, weights = [], [], []
    for _, _, x, o, j, yo, yj, _ in groups:
        mask = o != j
        if mask.any():
            data.append(x[mask])
            response.append((yj - yo)[mask])
            weights.append(np.ones(int(mask.sum())) / mask.size)
    x = np.concatenate(data).astype(np.float64)
    y = np.concatenate(response)
    w = np.concatenate(weights)
    w /= w.sum()
    mu = w @ x
    scale = np.maximum(np.sqrt(w @ ((x - mu) ** 2)), 1e-5)
    z = np.column_stack((np.ones(len(x)), (x - mu) / scale))
    regularizer = np.diag([0.] + [RIDGE_PENALTY] * x.shape[1])
    beta = np.linalg.solve(z.T @ (w[:, None] * z) + regularizer, z.T @ (w * y))
    write(output, dict(expert_sha256="37fc573a287b333b96892086ff4bf3d28a1247bd9340bd44c4041c698aeabf33",
        fit_prediction_manifest_sha256=sha(predictions / "predictions.json"), centers=mu.tolist(),
        scales=scale.tolist(), beta=beta.tolist(), ridge=RIDGE_PENALTY, min_expected_gain=MIN_EXPECTED_GAIN,
        calibration_disagreement_entries=len(y), calibration_entries_with_target_difference=int(np.count_nonzero(y))))


def evaluate(index, predictions, gate, output):
    params = json.loads(gate.read_bytes())
    weights = np.array(params["beta"])
    results = defaultdict(lambda: defaultdict(list))
    house_set = set()
    for row, cols, x, o, j, yo, yj, origin in records(index, predictions):
        house_set.add(row["house"])
        expected = weights[0] + ((x - params["centers"]) / params["scales"]) @ weights[1:]
        use_j = (o != j) & (expected > params["min_expected_gain"])
        picked = np.where(use_j, yj, yo)
        for name, mask in [("official_six", cols < 6), ("derived_ten", cols >= 6)]:
            if not mask.any():
                continue
            vals = [((s - origin)[:, :, mask]).mean() for s in (yo, yj, picked)]
            vals += [m[:, :, mask].mean() for m in (use_j, picked > yo, picked < yo)]
            results[name][row["house"]].append(vals)
    assert len(house_set) == 3
    output_data = {}
    for group, houses in results.items():
        by_house = {h: np.mean(a, 0).tolist() for h, a in houses.items()}
        av = np.mean(list(by_house.values()), 0)
        output_data[group] = dict(columns=["observed_gain", "joint_gain", "switch_gain", "used_different_joint_action",
            "switch_helped_over_observed", "switch_hurt_over_observed"], by_house=by_house, average=av.tolist(),
            switch_minus_observed=float(av[2] - av[0]))
    write(output, dict(gate_sha256=sha(gate), blind_predictions_sha256=sha(predictions / "predictions.json"),
                       groups=output_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fit", "evaluate"])
    for name in ("index", "predictions", "gate", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.mode == "fit":
        fit(args.index, args.predictions, args.gate)
    else:
        evaluate(args.index, args.predictions, args.gate, args.output)
