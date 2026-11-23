"""Score plans from actual goal text and RGB history. Teacher labels are absent."""
import math

import torch

from features import forecast_inputs
from grounded_route import evidence_fields, jepa_fields


@torch.no_grad()
def score_plans(forecasters, grounder, heads, observed_crops, observed_local, heading_bins, actions, text, mode):
    """Return four independently trained probability matrices [candidates, queries]."""
    if actions.ndim != 2 or actions.shape[1] not in (4, 12, 24):
        raise ValueError("plans must share one supported physical budget")
    if torch.any((actions < 0) | (actions > 2)):
        raise ValueError("action 0=forward, 1=left, 2=right")
    bins = [int(x) for x in heading_bins]
    if len(set(bins)) != len(bins) or 0 not in bins or any(i < 0 or i >= 12 for i in bins):
        raise ValueError("provide distinct actually observed heading bins including the current view")
    if len(bins) != len(observed_crops) or observed_local.shape != (len(bins), 49, 1024):
        raise ValueError("expected 49 observed regional RN50 tokens per retained view")
    if observed_crops.shape[1:] != (3, 1024) or text.ndim != 2 or text.shape[-1] != 1024:
        raise ValueError("expected three regional crops and aligned RN50 text")
    if mode not in ("full", "sparse"):
        raise ValueError("supply the supported mode from heading_coverage.choose_mode")
    heading = observed_crops.new_tensor(bins) * (math.pi / 6)
    # The private model fits on FP16-serialized local caches, computes in FP32.
    code = ((observed_local - forecasters[0].center) @ forecasters[0].basis).half().float()
    anchor_logits = (observed_local @ grounder.anchors.T).half().float()
    query_logits = (observed_local @ text.T).half().float()
    visible, near = grounder(code, anchor_logits, query_logits, text)
    p, nearest, _ = forecast_inputs(forecasters,mode,observed_crops,heading,actions)
    dense, geom = evidence_fields(visible,near,heading,actions)
    forecast = jepa_fields(p,nearest,text,grounder.anchors)
    return {key: head(dense,forecast,geom)[0].sigmoid() for key,head in heads.items()}


def select_route(scores, original_index, minimum_advantage=.01):
    """Select one different plan only when its probability strictly exceeds the margin."""
    if scores.ndim != 1 or not scores.numel() or not torch.isfinite(scores).all():
        raise ValueError("one score is required for each distinct legal plan")
    if not 0 <= original_index < scores.numel():
        raise ValueError("invalid reference index")
    candidate=int(scores.argmax())
    return candidate if float(scores[candidate]-scores[original_index])>minimum_advantage else original_index
