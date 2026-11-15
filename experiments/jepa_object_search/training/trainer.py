"""Compare a direct frontier scorer with the same head trained to predict visual features."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Decision:
    sid: str
    source: str
    house: str
    label_kind: str
    query: str
    online: dict
    future: torch.Tensor
    gain: torch.Tensor
    presence: torch.Tensor


def query_tokens(query):
    """Stable word buckets for known training queries; no language pretraining assumed."""
    words = re.findall(r"[^\W_]+", query.casefold())
    if not words:
        raise ValueError("Empty goal query")
    return [int.from_bytes(hashlib.sha256(w.encode()).digest()[:4], "little") % 2048 for w in words]


def read_pack(directory, text_vectors=None):
    directory = directory.resolve(strict=True)
    index = (directory / "index.jsonl").read_bytes()
    receipt = directory / "receipt.json"
    if receipt.is_file() and json.loads(receipt.read_text())["index_sha256"] != hashlib.sha256(index).hexdigest():
        raise ValueError("Pack manifest checksum changed")
    digest = hashlib.sha256(index)
    splits = {"train": [], "dev": []}
    ids, files = set(), set()
    for line in index.splitlines():
        row = json.loads(line)
        sid, source, house, split, query, name, label = (
            row[key] for key in ("id", "source", "house", "split", "query", "path", "label_kind"))
        path = (directory / name).resolve(strict=True)
        if not path.is_relative_to(directory) or path in files or sid in ids:
            raise ValueError("Duplicate record or file outside the pack")
        if not all(isinstance(x, str) and x.strip() for x in (sid, source, house, query, label)):
            raise ValueError("Invalid record metadata")
        if split not in splits:
            raise ValueError("Only house-disjoint train/dev inputs are supported")
        files.add(path)
        ids.add(sid)
        fingerprint = hashlib.sha256(path.read_bytes())
        if "sha256" in row and row["sha256"] != fingerprint.hexdigest():
            raise ValueError(f"State fingerprint changed: {sid}")
        digest.update(fingerprint.digest())
        with np.load(path, allow_pickle=False) as z:
            context, future, xy, actions, gain, report, report_xy, known, step, turn = (
                torch.from_numpy(np.array(z[k], copy=True)) for k in
                ("context", "future", "xy", "actions", "gain", "memory", "memory_xy", "has_memory", "step_m", "turn_deg"))
            context = context.float()
            future = future.float()
            if context.ndim == 1:
                context = context.unsqueeze(0)
            if future.ndim == 2:
                future = future.unsqueeze(1)
            if actions.ndim != 2:
                raise ValueError(f"Invalid decision-time action sequences: {sid}")
            n, horizon = actions.shape
            presence = (torch.from_numpy(np.array(z["region_presence"], copy=True)).float()
                        if "region_presence" in z else torch.full((n,), -1.))
        if not (context.ndim == 2 and context.size(1) == 768 and context.size(0) >= 1
                and n >= 2 and horizon > 0 and tuple(future.shape) == (n, *context.shape)
                and xy.shape == (n, 2) and gain.shape == (n,) and presence.shape == (n,)
                and report.shape == (768,) and report_xy.shape == (2,)
                and known.numel() == step.numel() == turn.numel() == 1):
            raise ValueError(f"Invalid decision/teacher dimensions: {sid}")
        if actions.is_floating_point() or actions.dtype == torch.bool or actions.min() < 0 or actions.max() > 2:
            raise ValueError(f"Invalid nominal action vocabulary: {sid}")
        if not all(t.isfinite().all() for t in (context, future, xy, gain, report, report_xy, step, turn)):
            raise ValueError(f"Nonfinite pack values: {sid}")
        if known.item() not in (0, 1) or step.item() <= 0 or not 0 < turn.item() <= 180 or gain.min() < 0:
            raise ValueError(f"Invalid flags, gain or motion units: {sid}")
        if not torch.isin(presence, presence.new_tensor([-1, 0, 1])).all():
            raise ValueError(f"Invalid region-presence mask: {sid}")
        # Unavailable report features cannot act as a source of privileged input.
        memory_valid = float(known.item())
        online = dict(context=context, xy=xy.float(), actions=actions.long(),
                      memory=report.float() * memory_valid, memory_xy=report_xy.float() * memory_valid,
                      has_memory=torch.tensor(memory_valid), step_m=step.float().reshape(()),
                      turn_rad=torch.deg2rad(turn.float()).reshape(()),
                      query=torch.tensor(query_tokens(query)) if text_vectors is None else text_vectors[query])
        splits[split].append(Decision(sid, source, house, label, query, online, future, gain.float(), presence))
    houses = {split: {(r.source, r.house) for r in records} for split, records in splits.items()}
    if not all(houses.values()) or houses["train"] & houses["dev"]:
        raise ValueError("Empty split or training/development house overlap")
    return splits, digest.hexdigest()


class RouteHead(nn.Module):
    def __init__(self, center, basis, text_width=0):
        super().__init__()
        w = basis.shape[1]
        self.register_buffer("center", center.detach().clone())
        self.register_buffer("basis", basis.detach().clone())
        self.path_encoder = nn.GRU(3, 32, batch_first=True)
        self.transition = nn.Sequential(nn.Linear(w + 37, 96), nn.GELU(), nn.Linear(96, w))
        self.text_width = text_width
        self.query_encoder = nn.Linear(text_width, 24) if text_width else nn.Embedding(2048, 24)
        self.query_key = nn.Linear(24, w)
        self.scorer = nn.Sequential(nn.Linear(2*w + 24 + 3, 64), nn.GELU(), nn.Linear(64, 2))

    def project(self, feature):
        return (feature - self.center) @ self.basis

    def forward(self, *, context, xy, actions, memory, memory_xy, has_memory, step_m, turn_rad, query):
        n = len(xy)
        # Motion commands use the actual per-source metric step and rotation.
        motion = torch.stack([(actions == 0) * step_m, (actions == 1) * turn_rad,
                              (actions == 2) * turn_rad], dim=-1)
        _, state = self.path_encoder(motion)
        branch = torch.cat([state[0], xy], dim=-1)
        past = self.project(context)[None].expand(n, -1, -1)
        if past.size(1) == 8:
            coord = torch.cartesian_prod(*(past.new_tensor([-1.,1.]) for _ in range(3)))
        else:
            coord = past.new_zeros((past.size(1),3))
        conditioned = torch.cat([past, branch[:, None].expand(-1, past.size(1), -1),
                                 coord.expand(n,-1,-1)], dim=-1)
        predicted = past + self.transition(conditioned)
        encoded = self.query_encoder(query)
        goal = encoded if self.text_width else encoded.mean(0)
        weights = torch.softmax((predicted @ self.query_key(goal)) / math.sqrt(predicted.size(-1)), dim=1)
        view = (weights.unsqueeze(-1) * predicted).sum(1)
        memory_state = self.project(memory) * has_memory
        displacement = (xy - memory_xy) * has_memory
        features = torch.cat([view, memory_state.expand(n, -1), goal.expand(n, -1),
                              displacement, has_memory.expand(n, 1)], dim=1)
        output = self.scorer(features)
        return output[:, 0], predicted, output[:, 1]


def fit_basis(train):
    # Fit only to past frames of training houses. Future and dev observations are excluded.
    past = torch.cat([r.online["context"] for r in train])
    center = past.mean(0)
    if len(past) < 2:
        raise ValueError("At least two training observations are required")
    _, singular, vh = torch.linalg.svd(past - center, full_matrices=False)
    w = min(48, int((singular > singular[0] * 1e-6).sum().item()))
    if not w:
        raise ValueError("Training context features have no visual variation")
    return center, vh[:w].T.contiguous()


def move(online, device):
    return {key: tensor.to(device) for key, tensor in online.items()}


def losses(model, item, device, predictive):
    value, future, region = model(**move(item.online, device))
    gain = item.gain.to(device)
    # Targets can have different physical meanings across fixture kinds; ranking never crosses states.
    ordered = gain[:, None] > gain[None, :] + 1e-6
    delta = value[:, None] - value[None, :]
    ranking = F.softplus(-delta[ordered]).mean() if ordered.any() else value.sum() * 0
    label = item.presence.to(device)
    known = label >= 0
    if known.any():
        ranking = ranking + F.binary_cross_entropy_with_logits(region[known], label[known])
    if predictive:
        target = model.project(item.future.to(device)).detach()
        latent_loss = 1 - F.cosine_similarity(future, target, dim=-1).mean()
        return ranking + latent_loss
    return ranking


@torch.inference_mode()
def evaluate(model, records, device):
    buckets = {}
    for r in records:
        scores, _, region_logits = model(**move(r.online, device))
        scores = scores.cpu()
        tied_best = torch.isclose(scores, scores.max(), atol=1e-7, rtol=1e-6)
        achieved = r.gain[tied_best].mean().item()
        ordered = r.gain[:, None] > r.gain[None, :] + 1e-6
        delta = (scores[:, None] - scores[None, :])[ordered]
        accuracy = (float((delta > 1e-7).float().mean() + 0.5*(delta.abs() <= 1e-7).float().mean())
                    if len(delta) else None)
        key = (r.source, r.label_kind)
        valid = r.presence >= 0
        labels = r.presence[valid]
        region_errors = (region_logits.cpu().sigmoid()[valid] - labels).square()
        buckets.setdefault(key, []).append((r.house, achieved, float(r.gain.max()), accuracy,
                                             float(r.gain.mean()), int(labels.sum()), len(labels),
                                             float(region_errors.sum())))
    result = []
    for (source, label), rows in sorted(buckets.items()):
        acc = [r[3] for r in rows if r[3] is not None]
        n = sum(r[6] for r in rows)
        pos = sum(r[5] for r in rows)
        result.append(dict(source=source, label_kind=label, houses=len({r[0] for r in rows}),
                           states=len(rows), informative_states=len(acc),
                           mean_chosen_gain=float(np.mean([r[1] for r in rows])),
                           mean_uniform_gain=float(np.mean([r[4] for r in rows])),
                           mean_chosen_minus_uniform=float(np.mean([r[1]-r[4] for r in rows])),
                           mean_oracle_gap=float(np.mean([r[2]-r[1] for r in rows])),
                           observed_region_brier=sum(r[7] for r in rows)/n if n else None,
                           region_n_pos=pos, region_n_neg=n-pos,
                           paired_accuracy_informative=float(np.mean(acc)) if acc else None))
    return result


def describe(splits):
    return {name: {"states": len(rows), "branches": sum(len(r.gain) for r in rows),
                   "houses": len({(r.source,r.house) for r in rows}),
                   "source_states": dict(Counter(r.source for r in rows)),
                   "label_kinds": dict(Counter(r.label_kind for r in rows)),
                   "query_count": len({r.query for r in rows}),
                   "states_with_report_memory": sum(bool(r.online["has_memory"]) for r in rows),
                   "region_labeled_candidates": sum(int((r.presence >= 0).sum()) for r in rows),
                   "informative_states": sum(float(r.gain.max()-r.gain.min()) > 1e-6 for r in rows),
                   "visual_tokens_per_state": sorted({r.online["context"].size(0) for r in rows})}
            for name, rows in splits.items()}


def run(args):
    if args.epochs < 1 or args.batch < 1 or args.threads < 1:
        raise ValueError("Positive epoch, batch and thread counts required")
    if not args.check_only and (args.output is None or args.output.exists()):
        raise ValueError("Training requires a fresh --output directory")
    torch.set_num_threads(args.threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    text_vectors, text_info, text_width = None, {"method":"hashed-training-vocabulary"}, 0
    if args.text_cache:
        with np.load(args.text_cache, allow_pickle=False) as z:
            names = z["queries"].tolist()
            v = torch.from_numpy(z["vectors"]).float()
            if v.shape != (len(names),1024) or len(set(names))!=len(names) or not v.isfinite().all():
                raise ValueError("Invalid BGE-M3 goal embedding cache")
            text_vectors = dict(zip(names, v))
            text_width = v.size(1)
            text_info = json.loads(z["protocol"].item())
            text_info["cache_sha256"] = hashlib.sha256(args.text_cache.read_bytes()).hexdigest()
    splits, data_sha = read_pack(args.data, text_vectors)
    train, dev = splits["train"], splits["dev"]
    center, basis = fit_basis(train)
    receipt = dict(format="multi-source-frontier-v1", dataset_sha256=data_sha,
                   trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   split=describe(splits), seed=args.seed, epochs=args.epochs, batch=args.batch,
                   lr=0.001, latent_weight=1., basis_width=basis.shape[1], torch=str(torch.__version__),
                   text=text_info, device=args.device, arms={})
    device = torch.device(args.device)
    if args.check_only:
        model = RouteHead(center, basis, text_width).to(device).eval()
        with torch.inference_mode():
            for r in train + dev:
                if not torch.isfinite(losses(model, r, device, True)):
                    raise ArithmeticError("Nonfinite preflight loss")
        print(json.dumps({"preflight": "forward_only", **receipt}, ensure_ascii=False))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        stage = Path(tmp)/"run"
        stage.mkdir()
        for mode in ("direct", "jepa"):
            torch.manual_seed(args.seed)
            rng = random.Random(args.seed)
            model = RouteHead(center, basis, text_width).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=receipt["lr"])
            history = []
            for epoch in range(args.epochs):
                order = list(range(len(train)))
                rng.shuffle(order)
                total = 0.
                for start in range(0, len(order), args.batch):
                    optimizer.zero_grad(set_to_none=True)
                    indices = order[start:start+args.batch]
                    loss = torch.stack([losses(model, train[i], device, mode == "jepa") for i in indices]).mean()
                    if not torch.isfinite(loss):
                        raise ArithmeticError("Nonfinite training loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                    optimizer.step()
                    total += float(loss.detach()) * len(indices)
                history.append(total / len(train))
            model.eval()
            checkpoint = stage/f"{mode}.pt"
            torch.save({"format":receipt["format"], "mode":mode, "text_width":text_width, "text":text_info,
                        "state":{k:v.detach().cpu() for k,v in model.state_dict().items()}}, checkpoint)
            restored = load_checkpoint(checkpoint, device)
            result = evaluate(restored, dev, device)
            receipt["arms"][mode] = dict(dev=result, train_loss=history,
                                         num_parameters=sum(p.numel() for p in model.parameters()),
                                         checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            print(f"{mode}: {json.dumps(result)}", flush=True)
        (stage/"receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(args.output)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if payload["format"] != "multi-source-frontier-v1":
        raise ValueError("Unexpected checkpoint format")
    state = payload["state"]
    model = RouteHead(state["center"], state["basis"], payload["text_width"])
    model.load_state_dict(state)
    return model.to(device).eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Verify all records and forward passes without an optimizer or output")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8, help="Decision points per gradient update")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text-cache", type=Path, help="Frozen BGE-M3 query cache from text_cache.py")
    run(parser.parse_args())
