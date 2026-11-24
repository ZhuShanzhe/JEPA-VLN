"""Translate native, reachable frontier paths and measured camera rays."""
import math
import re
from functools import lru_cache

import numpy as np


def category(text):
    # Use only the task given to the policy. Attributes are still judged by MOST.
    rules = [(r"\b(?:pictures?|paintings?|artworks?|wall art|prints?)\b", "wall_art.n.01"),
             (r"\bclocks?\b", "clock.n.01"), (r"\bmirrors?\b", "mirror.n.01"),
             (r"\b(?:lamps?|light fixtures?)\b", "lamp.n.01"), (r"\bdoors?\b", None)]
    for pattern, word in rules:
        if re.search(pattern, text, re.I):
            return word
    return None


def plan_table(rows, native):
    ordered = sorted(rows, key=lambda row: row["id"] != native)
    ids, plans = [], []
    for row in ordered:
        plan = tuple((row.get("nominal_action_ids") or [])[:24])
        if len(plan) == 24 and set(plan) <= {1, 2, 3} and plan not in plans:
            ids.append(row["id"])
            plans.append(plan)
    if not ids or ids[0] != native:
        return [], [], "native_plan_short"
    if len(ids) < 3:
        return ids, plans, "fewer_than_three_distinct_native_plans"
    return ids[:3], plans[:3], None


@lru_cache(16)
def sampling(height, width, pitch):
    if height != width or height < 64 or pitch not in (0, 1):
        raise ValueError("unsupported native camera geometry")
    axis = (np.arange(128) + .5 - 64) / 64
    x, y = np.meshgrid(axis, -axis)
    t = math.radians(-15 + 30 * pitch)
    z = np.sin(t) * y + np.cos(t)
    f = width / (2 * math.tan(math.radians(60)))
    u = f * x / z + (width - 1) / 2
    v = -f * (np.cos(t) * y - np.sin(t)) / z + (height - 1) / 2
    if min(u.min(), v.min()) < 0 or u.max() > width - 1 or v.max() > height - 1:
        raise ValueError("unobserved rays")
    ix, iy = u.astype(int), v.astype(int)
    return ix, iy, (u - ix)[..., None], (v - iy)[..., None]


def camera(rgb, pitch):
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[-1] != 3 or a.dtype != np.uint8:
        raise ValueError("native camera must deliver RGB bytes")
    x, y, fx, fy = sampling(*a.shape[:2], pitch)
    x1, y1 = np.minimum(x + 1, a.shape[1] - 1), np.minimum(y + 1, a.shape[0] - 1)
    return np.rint((1 - fy) * ((1-fx)*a[y, x] + fx*a[y, x1]) +
                   fy * ((1-fx)*a[y1, x] + fx*a[y1, x1])).astype(np.uint8)
