"""Train and select one route head from a local cache of frozen CLIP features."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from route_heads import HORIZONS, SemanticChange, SparseHead, route_logits, view_inputs


def ranking_loss(logits, gain):
    """Compare candidate routes within a state, separately for every target phrase."""
    prob = logits.sigmoid()
    gap = gain[:, None] - gain[None, :]
    predicted_gap = prob[:, None] - prob[None, :]
    weight = gap.clamp_min(0) * (gap > .005)
    return (weight * F.softplus(-predicted_gap / .12)).sum() / weight.sum().clamp_min(.05)


def partial_arc(state, n_views, mask_seed):
    # One-sided contiguous turns end at the current camera heading (bin 0).
    key = (str(mask_seed) + "|" + state["sid"]).encode()
    sign = 1 if hashlib.sha256(key).digest()[0] & 1 else -1
    bins = sorted({(sign * offset) % 12 for offset in range(n_views)})
    return state["context"][bins], torch.tensor(bins).float() * (torch.pi / 6)


def predict(model, state, n_views=None, mask_seed=23):
    if n_views is None:
        fields = state["full_fields"]
        return model(*fields), fields[0], None
    observed, yaw = partial_arc(state, n_views, mask_seed)
    return model(observed, yaw, state["actions"], state["step"], state["turn"])


def training_loss(model, state, n_views=None, mask_seed=23):
    predicted, _, missing = predict(model, state, n_views, mask_seed)
    teacher = state["future"]
    error = predicted - teacher
    cosine_error = 1 - (predicted * teacher).sum(-1)
    if missing is not None:
        cosine_error = cosine_error * (1 + .35 * missing[:, :, None])
    centered_error = (error - error.mean(0, keepdim=True)).square().sum(-1).mean()
    logits = route_logits(predicted, state["text"])
    return cosine_error.mean() + .5 * centered_error + .20 * ranking_loss(logits, state["gain"])


def prepare(row):
    """Check one state/horizon cache record; preserve ordering of candidates and targets."""
    s = dict(row)
    action = s["actions"]
    if action.ndim != 2 or action.shape[0] < 2 or action.shape[1] not in HORIZONS:
        raise ValueError("need at least two routes at a trained horizon")
    n_routes = action.shape[0]
    n_queries = len(s["queries"])
    if n_queries < 1 or s["context"].shape != (12, 3, 1024):
        raise ValueError("invalid target list or panorama")
    if s["future"].shape != (n_routes, 2, 3, 1024) or s["gain"].shape != (n_routes, n_queries):
        raise ValueError("teacher/gain does not match candidate ordering")
    if s["text"].shape != (n_queries, 1024) or not torch.isin(action, torch.tensor([0, 1, 2])).all():
        raise ValueError("invalid CLIP text features or planned action IDs")
    if not all(torch.isfinite(s[k]).all() for k in ("text", "context", "future", "gain")):
        raise ValueError("invalid numeric input")
    if not ((s["gain"] >= 0) & (s["gain"] <= 1)).all():
        raise ValueError("expected a visibility proxy in [0, 1]")
    s["actions"] = action.long()
    s["gain"] = s["gain"].float()
    s["text"] = s["text"].float()
    s["context"] = F.normalize(s["context"].float(), dim=-1)
    s["future"] = F.normalize(s["future"].float(), dim=-1)
    yaw = s["yaw"].float()
    expected = torch.arange(12).float() * (torch.pi / 6)
    if yaw.shape != (12,) or not torch.allclose(yaw, expected, atol=1e-5, rtol=0):
        raise ValueError("sparse training expects 12 observed headings in ascending 30-degree steps")
    s["full_fields"] = view_inputs(s["context"], yaw, s["actions"], s["step"], s["turn"])
    return s


def gain_over_reference(prob, gain):
    # Treat each candidate in turn as the reference and average its possible improvement.
    best = prob.argmax()
    index = torch.arange(len(prob))
    revised = torch.where(prob[best] - prob > .01, best, index)
    return float((gain[revised] - gain).mean())


@torch.inference_mode()
def calibration_score(model, states, sparse):
    houses = defaultdict(list)
    for s in states:
        unique = sorted(set(s["queries"]))
        if len(unique) < 2:
            continue
        key = s["sid"], s["actions"].shape[1]
        rng = random.Random(int(hashlib.sha256(repr((23, key)).encode()).hexdigest(), 16))
        rng.shuffle(unique)
        alternate = dict(zip(unique, unique[1:] + unique[:1]))
        columns = {q: i for i, q in enumerate(s["queries"])}
        for n in ((1, 2, 3) if sparse else (None,)):
            future, nearest, missing = predict(model, s, n)
            if sparse and not missing.any():
                continue
            prob = route_logits(future, s["text"]).sigmoid()
            direct = route_logits(nearest, s["text"]).sigmoid()
            for i, q in enumerate(s["queries"]):
                gain = s["gain"][:, i]
                houses[s["house"]].append((
                    gain_over_reference(prob[:, i], gain),
                    gain_over_reference(prob[:, columns[alternate[q]]], gain),
                    gain_over_reference(direct[:, i], gain),
                ))
    if not houses:
        raise ValueError("no same-start distinct queries available for the calibration criterion")
    by_house = [[sum(row[i] for row in values)/len(values) for i in range(3)]
                for values in houses.values()]
    correct, wrong, direct = [sum(row[i] for row in by_house)/len(by_house) for i in range(3)]
    return correct - max(0., wrong, direct) if sparse else 2 * correct - wrong


def fit(cache_path, output, full_weights=None):
    if output.exists():
        raise FileExistsError(output)
    seed = 23
    rng = random.Random(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cache = torch.load(cache_path, weights_only=True, map_location="cpu")
    valid = {"train", "calibration", "check"}
    if any(r["split"] not in valid for r in cache):
        raise ValueError("unknown split label")
    by_house = defaultdict(set)
    for r in cache:
        by_house[r["house"]].add(r["split"])
    if any(len(splits) != 1 for splits in by_house.values()):
        raise ValueError("a house is shared across splits")
    key = lambda r: (r["sid"], r["actions"].shape[1])
    if len({key(r) for r in cache}) != len(cache):
        raise ValueError("group all targets from the same state and horizon into one record")
    train = [prepare(r) for r in sorted(cache, key=key) if r["split"] == "train"]
    calibration = [prepare(r) for r in sorted(cache, key=key) if r["split"] == "calibration"]
    if not train or not calibration:
        raise ValueError("train and calibration records are required")
    sparse = full_weights is not None
    if sparse:
        state = torch.load(full_weights, weights_only=True, map_location="cpu")
        base = SemanticChange(state["center"], state["basis"])
        base.load_state_dict(state)
        model = SparseHead(base)
        epochs, batch_size, lr = 9, 10, .0005
    else:
        contexts = {s["sid"]: s["context"].flatten(0, 1) for s in train}
        patches = torch.cat(list(contexts.values()))
        if len(patches) < 72:
            raise ValueError("PCA needs at least 72 observed crop embeddings from distinct train starts")
        center = patches.mean(0)
        _, _, vectors = torch.pca_lowrank(patches - center, q=72, center=False, niter=3)
        model = SemanticChange(center, vectors[:, :64].contiguous())
        epochs, batch_size, lr = 12, 12, .001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_score = -float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        rng.shuffle(train)
        model.train()
        for start in range(0, len(train), batch_size):
            batch = train[start:start+batch_size]
            loss = torch.stack([training_loss(model, s, rng.choice((1, 2, 3)) if sparse else None, epoch)
                                for s in batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
            optimizer.step()
        if epoch % 3 == 0:
            score = calibration_score(model.eval(), calibration, sparse)
            history.append(dict(epoch=epoch, calibration_criterion=score))
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best = copy.deepcopy(model.state_dict())
    with output.open("xb") as fp:
        torch.save(best, fp)
    return dict(selected_epoch=best_epoch, seed=seed, epochs=epochs, batch=batch_size,
                states=len(train), calibration_states=len(calibration), history=history)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--full-weights", type=Path, help="initialize V6 from a previously selected V5")
    args = p.parse_args()
    print(json.dumps(fit(args.cache, args.output, args.full_weights), indent=2, allow_nan=False))
