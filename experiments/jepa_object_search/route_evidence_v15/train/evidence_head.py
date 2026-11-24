"""Forecast evidence of a previously invisible object, conditional on a route."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class EvidenceNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.visual = nn.Sequential(nn.Linear(3*dim+8, 144), nn.LayerNorm(144), nn.GELU())
        self.goal = nn.Linear(dim, 144)
        self.fuse = nn.Sequential(nn.Linear(288, 128), nn.LayerNorm(128), nn.GELU())
        self.innovation = nn.Linear(128, dim)
        self.hazard = nn.Sequential(nn.Linear(128+3, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, routes, text):
        goal = text[:, None, None, :]
        visual = self.visual(routes)
        hidden = self.fuse(torch.cat((visual, visual*self.goal(goal)), dim=-1))
        departure = F.normalize(routes[..., :self.dim], dim=-1)
        prediction = F.normalize(departure + self.innovation(hidden), dim=-1)
        change = prediction - departure
        cues = torch.stack(((prediction*goal).sum(-1), (change*goal).sum(-1),
                            change.square().sum(-1)), dim=-1)
        hazard = self.hazard(torch.cat((hidden, cues),dim=-1)).squeeze(-1)
        # Complement of survival over four ordered physical observations.
        probability = -torch.expm1(F.logsigmoid(-hazard).sum(-1))
        return probability, hazard, prediction


class EvidencePredictor:
    def __init__(self, project):
        folder = Path(project)/"jepa_evidence_v15/train"
        manifest = json.loads((folder/"selection.json").read_text())
        path = folder/"frozen.pt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("evidence checkpoint does not match the selected model")
        self.net = EvidenceNet(manifest["dim"])
        self.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.net.eval()

    @torch.inference_mode()
    def predict(self, model, views, target, plans):
        plans = np.asarray(plans)
        if plans.ndim != 2 or plans.shape[1] != 24 or not np.isin(plans, [0,1,2]).all():
            raise ValueError("expected 24 planned native motion primitives in encoder coordinates")
        text = model.encoder.words([target])
        ctx, _ = model.encoder.encode(np.asarray(views), text)
        text = F.normalize(text @ model.encoder.proj, dim=-1).float().cpu()
        routes = model.old.route_inputs(ctx.astype(np.float32), plans)
        prob = self.net(torch.from_numpy(routes)[None], text)[0][0].numpy()
        if not np.isfinite(prob).all():
            raise ValueError("nonfinite evidence probabilities")
        return prob
