"""Inference for the HSSD-v2 pilot. All dimensions describe decision-time data."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class FrontierRanker(nn.Module):
    def __init__(self, goal_names, basis_dim=32):
        super().__init__()
        self.goal_names = tuple(goal_names)
        self.register_buffer("center", torch.zeros(768))
        self.register_buffer("basis", torch.zeros(768, basis_dim))
        self.register_buffer("scales", torch.ones(basis_dim))
        self.transition = nn.Sequential(
            nn.Linear(basis_dim + 2 + 24 * 3, 96), nn.GELU(), nn.Linear(96, basis_dim)
        )
        self.goal = nn.Embedding(len(goal_names), 12)
        self.value = nn.Sequential(nn.Linear(basis_dim + 12, 32), nn.GELU(), nn.Linear(32, 1))

    def project(self, features):
        return ((features - self.center) @ self.basis) / self.scales

    def forward(self, context, xy, actions, goal_id):
        ctx = self.project(context.reshape(1, 768)).expand(len(xy), -1)
        # Positions are expressed in metres in the start frame, not world coordinates.
        action_vec = F.one_hot(actions, num_classes=3).float().reshape(len(xy), 72)
        change = self.transition(torch.cat([ctx, xy / 5., action_vec], dim=1))
        future = ctx + change
        query = self.goal(goal_id.reshape(1)).expand(len(xy), -1)
        return self.value(torch.cat([future, query], dim=1)).squeeze(1), future

    @torch.inference_mode()
    def score(self, context_embedding, candidate_xy, planned_actions, target_synset):
        """Scores in candidate order. Inputs contain no future observations or labels."""
        device = self.center.device
        context = torch.as_tensor(context_embedding, device=device, dtype=torch.float32)
        xy = torch.as_tensor(candidate_xy, device=device, dtype=torch.float32)
        actions = torch.as_tensor(planned_actions, device=device)
        if context.shape != (768,) or xy.ndim != 2 or xy.shape[1] != 2 or not len(xy):
            raise ValueError("Expected a 768-vector and a nonempty (N,2) candidate matrix")
        if actions.shape != (len(xy), 24) or actions.dtype == torch.bool or actions.is_floating_point():
            raise ValueError("Expected (N,24) integer plan")
        if not torch.isfinite(context).all() or not torch.isfinite(xy).all():
            raise ValueError("Observations and positions must be finite")
        if actions.min() < 0 or actions.max() > 2:
            raise ValueError("Unknown action, expected 0=forward, 1=left, 2=right")
        if target_synset not in self.goal_names:
            raise ValueError(f"Unknown v2 pilot category: {target_synset}")
        goal_id = torch.tensor(self.goal_names.index(target_synset), device=device)
        logits, _ = self.forward(context, xy, actions.long(), goal_id)
        scores = torch.sigmoid(logits)
        if not torch.isfinite(scores).all():
            raise ArithmeticError("Invalid ranker scores")
        return scores.cpu().tolist()


def load_ranker(checkpoint):
    bundle = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if bundle["format"] != "hssd-visible-v2/v1":
        raise ValueError("Unsupported checkpoint format")
    model = FrontierRanker(bundle["goal_names"], bundle["basis_dim"])
    model.load_state_dict(bundle["state"], strict=True)
    return model.eval()
