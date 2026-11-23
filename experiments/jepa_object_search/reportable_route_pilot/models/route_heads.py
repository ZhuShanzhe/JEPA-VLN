"""Goal-conditioned route ranking from frozen, text-aligned visual features."""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


HORIZONS = (4, 12, 24)


def view_inputs(observed, headings, actions, step_m=0.25, turn_rad=math.pi / 6):
    """Retrieve actual past views closest in yaw to the two planned future views."""
    yaw = ((actions == 1).float() - (actions == 2).float()).cumsum(1) * turn_rad
    end = [actions.shape[1] // 2 - 1, actions.shape[1] - 1]
    forward = (actions == 0) * step_m
    x = (forward * yaw.cos()).cumsum(1)[:, end]
    y = (forward * yaw.sin()).cumsum(1)[:, end]
    facing = yaw[:, end]
    relative = facing[:, :, None] - headings
    nearest = relative.cos().argmax(-1)
    history = observed[:, 0].mean(0).expand(len(actions), 2, -1)
    horizon = facing.new_tensor([i + 1 for i in end]).expand(len(actions), 2) / 24
    motion = torch.stack([x / 6, y / 6, facing.cos(), facing.sin(), horizon], -1)
    return observed[nearest], history, motion


class SemanticChange(nn.Module):
    """V5. Predict local future image embeddings as residuals in a train-only PCA basis."""

    def __init__(self, center, basis):
        super().__init__()
        self.register_buffer("center", center)
        self.register_buffer("basis", basis)
        dim = basis.shape[1]
        self.delta = nn.Sequential(nn.Linear(2 * dim + 5, 96), nn.GELU(), nn.Linear(96, dim))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, directional, global_scene, motion):
        patches = (directional - self.center) @ self.basis
        history = (global_scene - self.center) @ self.basis
        fields = [patches, history[..., None, :].expand(-1, -1, 3, -1),
                  motion[..., None, :].expand(-1, -1, 3, -1)]
        correction = .25 * torch.tanh(self.delta(torch.cat(fields, -1))) @ self.basis.T
        return F.normalize(directional + correction, dim=-1)


class SparseHead(nn.Module):
    """V6. Predict from a partial observed arc, carrying angular uncertainty explicitly."""

    def __init__(self, full_head):
        super().__init__()
        self.base = full_head
        dim = full_head.basis.shape[1]
        self.unknown = nn.Sequential(nn.Linear(2 * dim + 10, 96), nn.GELU(), nn.Linear(96, dim))
        nn.init.zeros_(self.unknown[-1].weight)
        nn.init.zeros_(self.unknown[-1].bias)

    def forward(self, observed, headings, actions, step_m=0.25, turn_rad=math.pi / 6):
        yaw = ((actions == 1).float() - (actions == 2).float()).cumsum(1) * turn_rad
        facing = yaw[:, [actions.shape[1] // 2 - 1, actions.shape[1] - 1]]
        diff = facing[:, :, None] - headings
        closest = diff.cos().argmax(-1)
        err = diff.gather(-1, closest.unsqueeze(-1)).squeeze(-1)
        nearest, scene, motion = view_inputs(observed, headings, actions, step_m, turn_rad)
        weights = (2 * diff.cos()).softmax(-1)
        pooled = torch.einsum("btv,vcd->btcd", weights, observed)
        missing = (err.cos() < math.cos(math.radians(30.0001))).float()
        edge = torch.where(err.sin()[:, :, None] > 0, nearest[:, :, 1], nearest[:, :, 2])
        prior = F.normalize(nearest + .3 * err.sin().abs()[:, :, None, None] * edge[:, :, None], dim=-1)
        base = self.base(prior, scene, motion)
        pooled_code = (pooled - self.base.center) @ self.base.basis
        nearest_code = (nearest - self.base.center) @ self.base.basis
        extra = torch.stack([err.cos(), err.sin(), missing, torch.full_like(err, len(headings) / 12),
                             weights.amax(-1)], -1)
        local = torch.cat([motion, extra], -1)[:, :, None].expand(-1, -1, 3, -1)
        fields = torch.cat([pooled_code, nearest_code, local], -1)
        residual = .25 * torch.tanh(self.unknown(fields)) @ self.base.basis.T
        return F.normalize(base + residual, dim=-1), nearest, missing


def route_logits(regions, text):
    """Score candidate future image features against original frozen CLIP text vectors."""
    sim = regions @ text.T
    sim = sim.reshape(len(regions), -1, len(text))
    return ((torch.logsumexp(24 * sim, dim=1) - math.log(6)) / 24 - .22) * 20


def load_heads(full_file, sparse_file):
    """Read two weight-only state dictionaries into separate evaluation-mode CPU modules."""
    full = torch.load(Path(full_file), weights_only=True, map_location="cpu")
    sparse = torch.load(Path(sparse_file), weights_only=True, map_location="cpu")
    v5 = SemanticChange(full["center"], full["basis"])
    v5.load_state_dict(full)
    v6 = SparseHead(SemanticChange(sparse["base.center"], sparse["base.basis"]))
    v6.load_state_dict(sparse)
    return v5.eval(), v6.eval()


def compare_plans(plans, original):
    """Pair alternatives with the original at the longest horizon shared by both."""
    reference = plans[original] or []
    for plan in plans:
        if plan and any(a not in (1, 2, 3) for a in plan):
            raise ValueError("host action codes must be 1/2/3")
    pairs = []
    for index, plan in enumerate(plans):
        if index == original or not plan:
            continue
        common = [h for h in HORIZONS if h <= min(len(reference), len(plan))]
        if not common:
            continue
        h = max(common)
        ref, alt = list(reference[:h]), list(plan[:h])
        if ref == alt:
            continue
        split = next(i for i in range(h) if ref[i] != alt[i])
        pairs.append(dict(index=index, horizon=h, divergence=split, ref=[a-1 for a in ref],
                          alt=[a-1 for a in alt]))
    return pairs


def sector_trace(actions):
    yaw = 0
    bins = []
    for action in actions:
        yaw = (yaw + int(action == 1) - int(action == 2)) % 12
        bins.append(yaw)
    return [bins[len(bins)//2-1], bins[-1]]


def choose_mode(pairs, observations, pitch):
    """observations holds (relative heading bin, step seen), already filtered by distance."""
    info = dict(observations)
    bins = sorted(info)
    if pitch not in (0, 1) or not pairs or 0 not in info:
        return "bypass", [], []
    full = []
    if len(bins) >= 3:
        for pair in pairs:
            required = sector_trace(pair["ref"]) + sector_trace(pair["alt"])
            if all(min(abs((a-b+6) % 12 - 6) for b in bins) <= 1 for a in required):
                full.append(pair)
    if full:
        return "full", full, bins
    chains = []
    for sign in (1, -1):
        chain = [0]
        for offset in (1, 2):
            b = (sign * offset) % 12
            if b not in info:
                break
            chain.append(b)
        chains.append(chain)
    selected = max(chains, key=lambda c: (len(c), sum(info[b] for b in c)))
    return "sparse", pairs, sorted(selected)


def choose_pair(comparisons, original, gate=.01):
    if not math.isfinite(gate) or gate < 0:
        raise ValueError("invalid gate")
    chosen, advantage = original, 0.
    for item in comparisons:
        ref, alt = item["reference_score"], item["alternative_score"]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in (ref, alt)):
            raise ValueError("invalid route score")
        delta = alt - ref
        if delta > gate and delta > advantage:
            chosen, advantage = item["index"], delta
    return chosen, advantage


@torch.inference_mode()
def rank_routes(heads, plans, original, views, steps_seen, text_vector, pitch=0):
    """Run the fixed comparison policy and its observed-view control in one forward pass per pair."""
    if set(views) != set(steps_seen):
        raise ValueError("each measured view needs its observation step")
    mode, pairs, bins = choose_mode(compare_plans(plans, original), list(steps_seen.items()), pitch)
    result = dict(mode=mode, used_bins=bins, selected=original, observed_selected=original,
                  comparisons=[], observed_comparisons=[])
    if mode == "bypass":
        return result
    full, sparse = heads
    device = full.center.device
    ctx = F.normalize(torch.stack([views[b] for b in bins]).to(device=device, dtype=torch.float32), dim=-1)
    if ctx.shape[1:] != (3, 1024) or text_vector.numel() != 1024:
        raise ValueError("expected 3x1024 observed crop features and one 1024-dimensional text vector")
    # Inputs may already be normalized and quantized by the frozen feature provider.
    text = text_vector.reshape(1, -1).to(device=device, dtype=torch.float32)
    headings = torch.tensor(bins, device=device).float() * math.pi / 6
    for pair in pairs:
        actions = torch.tensor([pair["ref"], pair["alt"]], device=device)
        if mode == "full":
            inp = view_inputs(ctx, headings, actions)
            forecast, nearest = full(*inp), inp[0]
        else:
            forecast, nearest, _ = sparse(ctx, headings, actions)
        for key, regions in (("comparisons", forecast), ("observed_comparisons", nearest)):
            ref, alt = route_logits(regions, text).sigmoid().flatten().tolist()
            result[key].append(dict(index=pair["index"], horizon=pair["horizon"],
                                    divergence=pair["divergence"], reference_score=ref, alternative_score=alt))
    result["selected"], _ = choose_pair(result["comparisons"], original)
    result["observed_selected"], _ = choose_pair(result["observed_comparisons"], original)
    return result
