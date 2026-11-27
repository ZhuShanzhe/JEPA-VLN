"""Score feasible object-relative camera moves using a learned future-view representation."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


def crop_features(crop, box, full_shape):
    """Inputs are the observed RGB crop and its box in the original image."""
    if crop.ndim != 3 or crop.shape[2] < 3 or min(crop.shape[:2]) < 2:
        raise ValueError("Invalid object crop")
    width, height = full_shape[1], full_shape[0]
    x, y, u, v = np.asarray(box, dtype=float) / [width, height, width, height]
    rgb = cv2.resize(np.ascontiguousarray(crop[:, :, :3]), (32, 32), interpolation=cv2.INTER_AREA).astype("f") / 255
    tile = rgb.reshape(4, 8, 4, 8, 3)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    strength = np.hypot(dx, dy)
    ori = (np.arctan2(dy, dx) % np.pi * (6 / np.pi)).astype(int) % 6
    hist = np.stack([(strength * (ori == i)).reshape(4, 8, 4, 8).mean((1, 3)) for i in range(6)], axis=-1)
    hist /= np.sqrt((hist * hist).sum(axis=-1, keepdims=True) + .025)
    size = np.array([max(u-x,.0001), max(v-y,.0001)])
    return np.r_[tile.mean((1, 3)).ravel(), tile.std((1, 3)).ravel(), hist.ravel(),
                 np.log(size), np.log(size[0]/size[1]), x, y, u, v].astype("f")


def camera_features(angle, ratio):
    a, r = np.broadcast_arrays(np.asarray(angle, dtype="f"), np.asarray(ratio, dtype="f"))
    if not np.isfinite(a).all() or not np.isfinite(r).all() or (r <= 0).any():
        raise ValueError("Expected finite object-relative angles and positive ranges")
    lr = np.log(np.clip(r, .25, 4.0))
    return np.stack([np.sin(a), np.cos(a), np.sin(2*a), np.cos(2*a), lr,
                     lr*np.sin(a), lr*np.cos(a)], axis=-1)


class ViewPredictor:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as ck:
            self.p = {k: ck[k] for k in ck.files}

    def forward(self, inputs):
        p = self.p
        x = np.clip((inputs-p["mean"])/p["std"], -8, 8)
        h = np.maximum(0, x@p["w1"].T + p["b1"])
        z = h@p["w2"].T + p["b2"]
        score = (np.tanh(z)@p["w3"].T+p["b3"]).ravel()
        return 1/(1+np.exp(-np.clip(score,-40,40))), z

    def score_views(self, rgb, box, angles_rad, range_ratios):
        rgb = np.asarray(rgb)
        box = np.asarray(box,dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("Expected a box in original image coordinates")
        x,y,u,v=np.clip(np.r_[np.floor(box[:2]),np.ceil(box[2:])],0,np.tile(rgb.shape[1::-1],2)).astype(int)
        feat=crop_features(rgb[y:v,x:u], (x,y,u,v), rgb.shape)
        motion=camera_features(angles_rad,range_ratios).reshape(-1,7)
        inputs=np.column_stack([np.broadcast_to(feat,(len(motion),len(feat))),motion])
        return self.forward(inputs)[0]

    def rank_adjustment(self, rgb, box, *, yaw_change_rad, range_ratio, context=None):
        return float(self.score_views(rgb,box,[yaw_change_rad],[range_ratio])[0])


@lru_cache(maxsize=1)
def load_predictor():
    return ViewPredictor(Path(__file__).resolve().parent/"weights/view_jepa.npz")
