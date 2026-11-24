"""Separate target-specific route choice, timing of discovery, and memory effects."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


def read_rows(path, methods):
    result = {name: {} for name in methods}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        name = row["method"]
        if name not in result or not row["quota"]:
            continue
        key = (row["house"], row["origin"], row["seed"], row["word"])
        if key in result[name]:
            raise ValueError("Duplicate query")
        result[name][key] = row
    if len({frozenset(values) for values in result.values()}) != 1:
        raise ValueError("Methods have different queries")
    return result


def behavior(records):
    starts = defaultdict(list)
    count = Counter()
    for key, row in records.items():
        starts[key[:3]].append(row["path"][0])
        if row["first_step"] is not None:
            segment = (row["first_step"] - 1) // 28 + 1
            count[segment] += 1
    comparable = [routes for routes in starts.values() if len(routes) >= 2]
    return {
        "present_query_seed_runs": len(records),
        "start_seed_groups_with_at_least_two_goals": len(comparable),
        "start_seed_groups_choosing_different_initial_directions": sum(len(set(rs)) > 1 for rs in comparable),
        "first_observation_by_28_step_segment": [count[t] for t in (1, 2, 3, 4)],
    }


def matched(kept, ablated):
    count = Counter()
    for key, row in kept.items():
        other = ablated[key]
        count["changed_path"] += row["path"] != other["path"]
        diff = row["found"][-1] - other["found"][-1]
        count["more_instances" if diff > 0 else "fewer_instances" if diff < 0 else "equal_instances"] += 1
    return {k: count[k] for k in ("changed_path", "more_instances", "equal_instances", "fewer_instances")}


def analyze(folder):
    long = read_rows(folder / "fresh_horizon_decisions.jsonl", ["jepa_horizon", "direct_horizon", "v11_baseline"])
    short = read_rows(folder / "fresh_decisions.jsonl", ["jepa_full", "jepa_without_scene_evidence"])
    layout = read_rows(folder / "fresh_layout_decisions.jsonl", ["jepa_horizon_layout", "jepa_horizon"])
    return {
        "unit": "matched_query_seed_or_matched_start_seed",
        "budgets": [28, 56, 84, 112],
        "policies": {k: behavior(rows) for k, rows in long.items()},
        "cue_evidence_on_vs_off_without_horizon": matched(short["jepa_full"], short["jepa_without_scene_evidence"]),
        "layout_update_on_vs_off_with_horizon": matched(layout["jepa_horizon_layout"], layout["jepa_horizon"]),
    }


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    (folder / "route_behavior.json").write_text(json.dumps(analyze(folder), indent=2))
