"""Freeze the four grounded experts' switch features using only observed inputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from features import H, forecast_inputs, mask_bins
from grounded_route import evidence_fields, jepa_fields
from route_heads import load_heads
from train_v8 import restore_heads

HISTORY_SIZES = (12, 1, 2, 3)
STATE_KEYS = {"sid", "house", "code", "anchor_sim", "ctx", "actions"}


def choose_all_references(scores, distinct):
    """Apply the 0.01 margin separately for every possible original route and query."""
    alternatives = np.where(distinct[:, :, None], scores[None, :, :], -np.inf)
    keep = np.arange(len(scores))[:, None]
    return np.where(alternatives.max(axis=1) - scores > .01, alternatives.argmax(axis=1), keep)


def switch_inputs(observed, joint, jepa, geometry, distinct):
    first = choose_all_references(observed, distinct)
    second = choose_all_references(joint, distinct)
    query = np.arange(observed.shape[1])[None, :]
    delta = [scores[second, query] - scores[first, query]
             for scores in (observed, joint, jepa, geometry)]
    x = np.stack(delta + [observed[first, query], joint[second, query]], axis=-1)
    return first, second, x


@torch.inference_mode()
def predict_state(state, anchors, ground, experts, predictors):
    if set(state) != STATE_KEYS:
        raise ValueError("expected sid/house plus observed code, anchor_sim, ctx and chosen candidate actions")
    code, sim, observed, actions = [state[key] for key in ("code", "anchor_sim", "ctx", "actions")]
    if code.shape!=(12,49,64) or sim.shape!=(12,49,len(anchors)) or observed.shape!=(12,3,1024):
        raise ValueError("unexpected real-view feature dimensions")
    if actions.ndim!=2 or actions.shape[1]!=24 or (actions<0).any() or (actions>2).any():
        raise ValueError("expected candidate plans with 24 motion actions, 0=F,1=L,2=R")
    # ctx is the original crop cache. Match the offline predictor's normalization once.
    observed = F.normalize(observed.float(), dim=-1)
    visible, near = ground(code.float(), sim.float(), sim.float(), anchors)
    parts = []
    for horizon in H:
        a = actions.long()[:, :horizon]
        distinct = (a[:,None]!=a[None,:]).any(-1).cpu().numpy()
        for count in HISTORY_SIZES:
            bins = mask_bins(state["sid"], count)
            headings = observed.new_tensor(bins) * math.pi / 6
            mode = "full" if count == 12 else "sparse"
            p, nearest, _ = forecast_inputs(predictors, mode, observed[bins], headings, a)
            d, geometry = evidence_fields(visible[bins], near[bins], headings, a)
            forecast = jepa_fields(p, nearest, anchors, anchors)
            scores = [experts[k](d, forecast, geometry)[0].sigmoid().numpy()
                      for k in ("observed","joint","jepa","geometry")]
            parts.append(switch_inputs(*scores, distinct))
    return tuple(map(np.stack, zip(*parts)))


def save_predictions(cache, checkpoint, full, sparse, destination):
    bundle = torch.load(cache, weights_only=True, map_location="cpu")
    if set(bundle)!={"anchors","queries","states"}: raise ValueError("unexpected cache fields")
    ground, experts, words = restore_heads(checkpoint)
    anchors = bundle["anchors"].float()
    if list(bundle["queries"])!=words or not torch.allclose(anchors,ground.anchors,rtol=1e-5,atol=1e-6):
        raise ValueError("expected the checkpoint's complete ordered training vocabulary")
    if not bundle["states"]: raise ValueError("empty prediction cache")
    if len({s["sid"] for s in bundle["states"]}) != len(bundle["states"]):
        raise ValueError("duplicate physical origin identifier")
    predictors = load_heads(full,sparse)
    destination.mkdir(exist_ok=False,parents=True)
    manifest=[]
    for state in bundle["states"]:
        o,j,x = predict_state(state,anchors,ground,experts,predictors)
        name=hashlib.sha256(state["sid"].encode()).hexdigest()+".npz"
        path=destination/name
        np.savez_compressed(path,words=words,observed=o,joint=j,x=x)
        manifest.append(dict(sid=state["sid"],house=state["house"],file=name,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    with (destination/"predictions.json").open("x",encoding="utf-8") as out:
        json.dump(manifest,out,ensure_ascii=False,indent=2,allow_nan=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("cache","checkpoint","full","sparse","predictions"):
        parser.add_argument("--"+arg,type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    save_predictions(args.cache,args.checkpoint,args.full,args.sparse,args.predictions)
