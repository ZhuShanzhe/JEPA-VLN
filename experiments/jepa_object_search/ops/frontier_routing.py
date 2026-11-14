"""Route an already-parsed navigation action. The host retains parsing and execution."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class FrontierChoice:
    original: object
    proposed: object
    executed: object
    reason: str


def choose_frontier(kind, original, legal_ids, score_candidates, *, apply=False):
    """Use the host's ordered legal IDs. Its original validation still follows this call."""
    def keep(reason):
        return FrontierChoice(original, original, original, reason)

    if kind != "explore":
        return keep("host_action")
    ids = tuple(legal_ids)
    if original not in ids or len(set(ids)) != len(ids):
        return keep("invalid_host_candidates")
    if len(ids) < 2:
        return keep("single_candidate")
    try:
        scores = tuple(float(score) for score in score_candidates(ids))
        if len(scores) != len(ids) or not all(isfinite(s) for s in scores):
            raise ValueError("Candidate score count or value is invalid")
    except Exception as exc:
        return keep("scoring_error/" + type(exc).__name__)
    best = max(scores)
    index = ids.index(original)
    proposed = original if scores[index] == best else ids[scores.index(best)]
    return FrontierChoice(original, proposed, proposed if apply else original,
                          "apply" if apply else "shadow")
