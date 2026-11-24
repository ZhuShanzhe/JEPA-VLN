"""Freeze a positive-scale hazard calibration, then score untouched HSSD houses."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from variable_evidence import VariableEvidence


@torch.inference_mode()
def read_logits(project, net, split):
    from train_compare import route_inputs
    base = project / ("jepa_route_v12" if split == "fresh2" else "jepa_hidden_v11")
    root = base / "models/cache/hssd" / split
    logits, hits, first, house = [], [], [], []
    parity = 0.
    from variable_evidence import feature_rows
    for path in sorted((root / "actor").glob("*.npz")):
        with np.load(path) as actor:
            plans, ctx = actor["actions"], actor["ctx"].astype(np.float32)
            x = torch.from_numpy(route_inputs(ctx, plans))
            features, taps = feature_rows(ctx, plans, route_inputs)
            parity = max(parity, float(np.max(np.abs(features.reshape(x.shape) - x.numpy()))))
            assert taps == [[6, 12, 18, 24]] * len(plans)
            text = torch.from_numpy(actor["text"].astype(np.float32))
            n = len(text)
            logits.extend(net(x[None].expand(n, -1, -1, -1), text)[1].numpy())
            house += [str(actor["house"])] * n
            if split == "calibration":
                with np.load(project / "jepa_route_v12/data/local_targets/calibration" / (str(actor["origin"]) + ".npz")) as labels:
                    assert labels["words"].tolist() == actor["words"].tolist()
                    first.extend(labels["first_tap"].T)
        with np.load(root / "truth" / path.name) as labels:
            hits.extend(labels["y"].T > 0)
    assert house and parity <= 1.e-5
    return np.array(logits), np.array(hits), np.asarray(first), np.array(house), parity


def calibrate(logits, first):
    tap = np.arange(4)
    mask = (first[..., None] < 0) | (tap <= first[..., None])
    event = tap == first[..., None]
    # Calibrate against observed at-risk samples; regularize sparse positives.
    x = torch.as_tensor(logits[mask], dtype=torch.float64)
    y = torch.as_tensor(event[mask], dtype=torch.float64)
    params = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([params], max_iter=60, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        z = params[0].exp() * x + params[1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) + .001 * params.square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    return params.detach()[0].exp().item(), params.detach()[1].item()


def save(path, data):
    content = json.dumps(data, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != content:
        raise FileExistsError(path)
    if not path.exists():
        path.write_text(content)


def main(project):
    v15 = project / "jepa_evidence_v15/train"
    sys.path[:0] = [str(v15), str(project / "jepa_hidden_v11/models")]
    from evidence_head import EvidencePredictor
    net = EvidencePredictor(project).net
    folder = project / "jepa_variable_v16/model"
    torch.set_num_threads(2)
    l, h, f, houses, _ = read_logits(project, net, "calibration")
    assert np.array_equal(f >= 0, h)
    s, b = calibrate(l, f)
    save(folder / "calibration.json", {
        "scale": s, "offset": b, "training_split": "calibration",
        "training_houses": sorted(set(houses)), "training_queries": len(h),
        "target": "first observed discovery at 6, 12, 18 or 24 actions, conditional on no previous discovery",
        "checkpoint_sha256": hashlib.sha256((v15 / "frozen.pt").read_bytes()).hexdigest(),
    })
    VariableEvidence(project)
    scores = {}
    for split in ("test", "fresh2"):
        logits, y, _, evaluation_houses, parity = read_logits(project, net, split)
        assert set(evaluation_houses).isdisjoint(houses)
        p = -np.expm1(-np.logaddexp(0., s * logits + b).sum(-1))
        p0 = -np.expm1(-np.logaddexp(0., logits).sum(-1))
        win = y[np.arange(len(y)), p.argmax(-1)]
        scores[split] = dict(queries=len(y), branches=y.shape[1],
            houses=len(set(evaluation_houses)), selected_discovery=int(win.sum()),
            random_discovery=float(y.mean()), brier=float(np.mean((p-y)**2)),
            original_brier=float(np.mean((p0-y)**2)), full_length_feature_max_abs_diff=parity)
    save(folder / "heldout_full.json", scores)
    print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
