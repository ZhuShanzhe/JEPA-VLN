"""Actor-only dispatch for a new viewpoint. Row indices refer to the offered candidates."""
import numpy as np


def shortest_per_view(views, eligible, lengths):
    ids = np.flatnonzero(eligible)
    representatives = []
    for view in np.unique(views[ids]):
        same = ids[views[ids] == view]
        representatives.append(int(same[np.argmin(lengths[same])]))
    return np.asarray(representatives, dtype=int)


def fallback(legacy, kind, views, admitted, lengths, move_budget, reach_scores, visible_scores):
    """Switch only on a real old abstention with all budgeted old candidates rejected."""
    fits = lengths <= move_budget
    if legacy is not None or np.any((kind == "old") & fits & admitted):
        return legacy
    reps = shortest_per_view(views, (kind == "alt") & fits & admitted, lengths)
    if not len(reps):
        return None
    merit = reach_scores[reps] * visible_scores[reps]
    if not np.isfinite(merit).all():
        raise ValueError("nonfinite actor predictions")
    # Deterministic tie break, favor fewer actions then candidate order.
    return int(reps[np.lexsort((reps, lengths[reps], -merit))[0]])
