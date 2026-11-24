"""Score real, ragged route prefixes with the frozen V15 latent-evidence model."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F


def real_prefixes(plans):
    result = []
    for plan in plans:
        actions = np.asarray(plan)
        if (actions.ndim != 1 or len(actions) < 1 or actions.dtype.kind not in "iu"
                or not np.isin(actions, (0, 1, 2)).all()):
            raise ValueError("each candidate must contain real forward/left/right actions")
        result.append(actions[:24])
    if not result:
        raise ValueError("at least one route is required for evidence prediction")
    return result


def feature_rows(ctx, plans, route_inputs):
    # There is no route padding: each geometric and visual input is computed at
    # the endpoint of a prefix actually proposed by the planner.
    slots = {}
    taps = []
    for i, plan in enumerate(plans):
        ends = list(range(6, len(plan) + 1, 6))
        if not ends or ends[-1] != len(plan):
            ends.append(len(plan))
        taps.append(ends)
        for j, t in enumerate(ends):
            slots.setdefault(t, []).append((i, j))
    features = [[None] * len(v) for v in taps]
    for t, positions in slots.items():
        chunk = np.stack([plans[i][:t] for i, _ in positions])
        endpoints = route_inputs(ctx, chunk)[:, -1, :]
        for index, (i, j) in enumerate(positions):
            features[i][j] = endpoints[index]
    return np.concatenate(features, axis=0), taps


def aggregate(logits, taps, scale, offset):
    """Interpolate survival for a final partial six-step observation interval."""
    probs, rates = [], []
    cursor = 0
    for ends in taps:
        widths = np.diff([0, *ends]) / 6.
        segment = logits[cursor:cursor + len(ends)]
        cursor += len(ends)
        log_survival = -np.sum(widths * np.logaddexp(0., scale * segment + offset))
        probs.append(float(-np.expm1(log_survival)))
        rates.append(float(-np.expm1(log_survival / widths.sum())))
    if cursor != len(logits) or not np.isfinite(probs + rates).all():
        raise ValueError("invalid evidence forecast")
    return probs, rates


class VariableEvidence:
    def __init__(self, project):
        project = Path(project)
        sys.path.insert(0, str(project / "jepa_evidence_v15/train"))
        from evidence_head import EvidencePredictor
        self.net = EvidencePredictor(project).net
        calibration = json.loads((project / "jepa_variable_v16/model/calibration.json").read_text())
        checkpoint = project / "jepa_evidence_v15/train/frozen.pt"
        if calibration["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError("calibrator and model checkpoints do not match")
        self.scale = float(calibration["scale"])
        self.offset = float(calibration["offset"])

    @torch.inference_mode()
    def predict(self, model, views, target, plans):
        plans = real_prefixes(plans)
        text = model.encoder.words([target])
        ctx, _ = model.encoder.encode(np.asarray(views), text)
        text = F.normalize(text @ model.encoder.proj, dim=-1).float().cpu()
        return self.predict_features(ctx.astype(np.float32), text, plans, model.old.route_inputs)

    @torch.inference_mode()
    def predict_features(self, ctx, text, plans, route_inputs):
        plans = real_prefixes(plans)
        rows, taps = feature_rows(ctx, plans, route_inputs)
        # The hazard MLP scores each endpoint independently; the route axis is
        # restored using the real prefix lengths after a single batched forward.
        _, hazard, _ = self.net(torch.from_numpy(rows)[None, None], text)
        p, rate = aggregate(hazard.flatten().numpy(), taps, self.scale, self.offset)
        return dict(evidence_probability=p, evidence_rate=rate,
                    evaluated_steps=[t[-1] for t in taps],
                    source="v15_hssd_calibrated_variable_prefix", tap_steps=taps)
