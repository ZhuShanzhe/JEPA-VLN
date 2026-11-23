"""Predict newly reportable target opportunities for equal-budget frontier routes."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from route_heads import route_logits, view_inputs


class ReportOpportunity(nn.Module):
    """Adapt frozen regional predictions without changing the original route model."""

    def __init__(self, center, basis):
        super().__init__()
        self.register_buffer("center", center.clone())
        self.register_buffer("basis", basis.clone())
        dim = basis.shape[1]
        self.residual = nn.Sequential(nn.Linear(2 * dim + 5, 48), nn.GELU(), nn.Linear(48, dim))
        self.readout = nn.Sequential(nn.Linear(2 * dim + 13, 48), nn.GELU(), nn.Linear(48, 1))
        for layer in (self.residual[-1], self.readout[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.bias = nn.Parameter(torch.tensor(-1.5))

    def forward(self, forecast, nearest, motion, text, observed_only=False):
        # forecast and nearest: branches x 2 future times x 3 observed crop types x 1024.
        # The original encoder and the forecast provider are frozen upstream.
        visual = (forecast - self.center) @ self.basis
        change = (forecast - nearest) @ self.basis
        movement = motion[:, :, None].expand(-1, -1, 3, -1)
        if observed_only:
            # The counterfactual removes both stages of future-feature prediction.
            # The text, known route motion and trained reportability readout stay fixed.
            predicted = nearest
        else:
            correction = self.residual(torch.cat([visual, change, movement], -1))
            predicted = F.normalize(forecast + .12 * correction.tanh() @ self.basis.T, dim=-1)
        aligned = F.normalize(text @ self.basis, dim=-1)
        local = F.normalize((predicted - self.center) @ self.basis, dim=-1)
        delta = (predicted - nearest) @ self.basis
        query = aligned[None, None, None]
        count = len(text)
        target = (predicted @ text.T)[..., None]
        earlier = (nearest @ text.T)[..., None]
        # Attach angular/motion evidence to measured target compatibility. These features
        # let a route approaching a target be valued differently from a static view.
        geom = movement[..., None, :].expand(-1, -1, -1, count, -1)
        patch = torch.eye(3, device=forecast.device, dtype=forecast.dtype)[None, None, :, None]
        patch = patch.expand(*target.shape[:-1], 3)
        evidence = torch.cat([local[..., None, :] * query, delta[..., None, :] * query,
                              target, target - earlier, geom * target, patch,
                              patch * (target - earlier)], -1)
        raw = self.readout(evidence).squeeze(-1)
        # Log-mean-exp across the predicted visual regions that may provide evidence.
        semantic = 20 * (target.squeeze(-1) - .22)
        logits = torch.logsumexp((semantic + raw).flatten(1, 2), dim=1) - math.log(6)
        return logits + self.bias, predicted


@torch.no_grad()
def forecast_inputs(heads, mode, observed, headings, actions, step=.25, turn=math.pi/6):
    nearest, scene, motion = view_inputs(observed, headings, actions, step, turn)
    if mode == "full":
        predicted = heads[0](nearest, scene, motion)
    elif mode == "sparse":
        predicted, nearest, _ = heads[1](observed, headings, actions, step, turn)
    else:
        raise ValueError(f"unknown view mode {mode!r}")
    return predicted, nearest, motion


def route_scores(model, forecast, nearest, motion, text, alpha, observed_only=False):
    scores, _ = model(forecast, nearest, motion, text, observed_only=observed_only)
    previous = route_logits(nearest if observed_only else forecast, text).sigmoid()
    return alpha * scores.sigmoid() + (1 - alpha) * previous


def load_v7(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    weights = state["head"]
    model = ReportOpportunity(weights["center"], weights["basis"])
    model.load_state_dict(weights)
    return model.eval(), state["alpha"]
