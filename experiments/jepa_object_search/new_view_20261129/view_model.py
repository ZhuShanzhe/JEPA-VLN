"""Visibility of the existing instance at a proposed new camera, conditional on reaching it."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch import nn

SKETCH = 24
SEED = 29117


@lru_cache(maxsize=1)
def projection():
    return np.random.default_rng(SEED).normal(size=(1024, SKETCH)).astype(np.float32)


def sketch(v):
    return np.asarray(v, np.float32) @ projection()


def current_inputs(context, pose, observed_map):
    """Whitelisted actor inputs, never images or boxes at the proposed camera."""
    p = np.asarray(pose, np.float32)
    ctx = np.asarray(context, np.float32)
    m = np.asarray(observed_map, np.float32)
    if p.shape != (len(ctx), 20) or ctx.shape[1:] != (2048,) or m.ndim != 2 or len(m) != len(ctx):
        raise ValueError("camera, source embeddings and observed map rows must align")
    z = np.maximum(p[:, 15], .12)
    scale = np.clip(p[:, 12] / z, .06, 8.)
    w, h = np.clip(p[:, 10] * scale, .002, 3), np.clip(p[:, 11] * scale, .002, 3)
    x = (1 + p[:, 13]) / 2
    y = (1 - p[:, 14]) / 2
    vis_w = np.maximum(0., np.minimum(1., x + w / 2) - np.maximum(0., x - w / 2))
    vis_h = np.maximum(0., np.minimum(1., y + h / 2) - np.maximum(0., y - h / 2))
    area = np.where(p[:, 15] > 0., vis_w * vis_h, 0.)
    log_area = np.log((area + .00005) / .01)
    evidence = np.column_stack([log_area, np.abs(p[:, 13:15]), scale, vis_w / w, vis_h / h,
                                area, np.linalg.norm(p[:, :2], axis=1)])
    x = np.column_stack((p, evidence, sketch(ctx[:, :1024]), sketch(ctx[:, 1024:]), m))
    if not np.isfinite(x).all():
        raise ValueError("non-finite inference input")
    return x.astype(np.float32), np.clip(log_area + np.log(.5), -6, 5).astype(np.float32)


class NewViewHead(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.torso = nn.Sequential(nn.Linear(dims, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 48), nn.GELU())
        self.future = nn.Linear(48, SKETCH * 2)
        self.visibility = nn.Sequential(nn.Linear(SKETCH * 2, 24), nn.GELU(), nn.Linear(24, 1))

    def forward(self, x, prior):
        future = self.future(self.torso(x))
        return prior + self.visibility(future).squeeze(1), future


class ViewPredictor:
    def __init__(self, saved):
        self.mean = saved["mean"]
        self.scale = saved["scale"]
        self.target_mean = saved["target_mean"]
        self.target_scale = saved["target_scale"]
        self.map_channels = int(saved["map_channels"])
        self.variant = saved["variant"]
        self.net = NewViewHead(len(self.mean))
        self.net.load_state_dict(saved["weights"])
        self.net.eval()

    @classmethod
    def load(cls, file):
        return cls(torch.load(Path(file), weights_only=True, map_location="cpu"))

    @torch.inference_mode()
    def predict(self, context, pose, observed_map):
        x, prior = current_inputs(context, pose, observed_map)
        if observed_map.shape[1] != self.map_channels:
            raise ValueError("observation map schema differs")
        x = torch.clamp((torch.from_numpy(x) - self.mean) / self.scale, -6, 6)
        if self.variant == "no_map":
            x[:, -self.map_channels:] = 0
        logits, delta = self.net(x, torch.from_numpy(prior))
        future = delta * self.target_scale + self.target_mean
        return {"p_visible": torch.sigmoid(logits).numpy(), "future_delta": future.numpy()}
