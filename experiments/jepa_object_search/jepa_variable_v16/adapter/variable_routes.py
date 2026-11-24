"""Reduce real MOST plans to distinct reachable movement branches."""
from numbers import Integral


# The native probe and trained head observe up to this many primitives per route.
# Shorter routes terminate naturally. No motion is appended.
FORECAST_STEPS = 24


def floor_trace(plan):
    """Headings of actual forward translations; in-place turns cover no floor."""
    heading = 0
    trace = []
    for action in plan:
        if action == 1:
            trace.append(heading)
        else:
            heading = (heading + (1 if action == 2 else -1)) % 12
    return tuple(trace)


def branches_differ(first, second):
    return any(a != b for a, b in zip(first, second))


def build_table(rows, original_id, available_steps):
    """Return native first, then all pairwise distinct routes in original order.

    available_steps excludes the four physical scan actions. If two alternatives
    have the same floor trace until one ends, retain the longer representative.
    MOST's selected path always has priority. No hindsight or simulator queries.
    """
    if not isinstance(available_steps, Integral) or isinstance(available_steps, bool):
        raise ValueError("expected integer post-scan action budget")
    horizon = min(FORECAST_STEPS, int(available_steps))
    if horizon < 1:
        return [], "insufficient_post_scan_budget"
    entries = []
    original = None
    seen_ids = set()
    for index, row in enumerate(rows):
        key = row["id"]
        if key in seen_ids:
            raise ValueError("duplicate native frontier id")
        seen_ids.add(key)
        raw = row.get("nominal_action_ids")
        if raw is None:
            continue
        plan = tuple(raw[:horizon])
        if any(isinstance(a, bool) or not isinstance(a, Integral) or a not in (1, 2, 3)
               for a in plan):
            continue
        trace = floor_trace(plan)
        if not trace:
            continue
        entry = (index, key, tuple(map(int, plan)), trace)
        if key == original_id:
            original = entry
        else:
            entries.append(entry)
    if original is None:
        return [], "native_has_no_translating_plan_in_horizon"
    retained = [original]
    for entry in sorted(entries, key=lambda e: (-len(e[3]), e[0])):
        if all(branches_differ(entry[3], previous[3]) for previous in retained):
            retained.append(entry)
    retained = [original, *sorted(retained[1:], key=lambda e: e[0])]
    result = [{"id": key, "plan": list(plan)} for _, key, plan, _ in retained]
    reason = None if len(result) >= 2 else "no_distinct_spatial_branch_in_horizon"
    return result, reason
