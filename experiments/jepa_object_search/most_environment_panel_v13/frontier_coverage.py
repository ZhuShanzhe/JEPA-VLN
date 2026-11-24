"""Estimate route-head input coverage using executed native frontier activations."""
from collections import Counter
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_bytes())


def limits(opts, native):
    if native not in opts:
        return set(), []
    orig = opts[native]
    distinct = {}
    for label, route in opts.items():
        if label == native:
            continue
        span = min(24, len(orig), len(route))
        if span < 6:
            continue
        k = next((i for i in range(span) if orig[i] != route[i]), None)
        if k is not None:
            distinct[route[:24]] = k + 1
    passed = set()
    for h in (12, 18, 24):
        if len(orig) < h:
            continue
        plans = {r[:h] for r in opts.values() if len(r) >= h}
        for count in (2, 3):
            if len(plans) >= count:
                passed.add(f"{count}_plans_{h}_steps")
    if distinct:
        passed.add("native_plus_one_distinct_plan_comparable_for_6_to_24_steps")
        if min(distinct.values()) <= 4:
            passed.add("native_plus_one_early_divergence_within_four_actions")
    return passed, sorted(distinct.values())


def run(pilot):
    cases = []
    for row in read(pilot / "panel.json")["rows"]:
        eid = row["episode_id"]
        ref = pilot / "comparison/accepted/second" / (eid + ".json")
        if not ref.exists():
            continue
        path = Path(read(ref)["result"]).parent / "agent/frontier_probe.jsonl"
        totals = Counter()
        events = []
        raw = [json.loads(s) for s in path.read_bytes().splitlines() if s]
        for event in raw:
            if event["event"] != "activation" or not event.get("activated"):
                continue
            opts = {}
            for item in event["candidates"]:
                plan = tuple(item["nominal_action_ids"])
                if plan and set(plan) <= {1, 2, 3}:
                    opts[item["id"]] = plan
            native = event["selected_id"]
            passed, offsets = limits(opts, native)
            totals["activated_native_frontiers"] += 1
            totals.update(passed)
            if offsets:
                events.append(offsets[0])
        cases.append(dict(ordinal=row["ordinal"], counts=dict(totals),
                          sum_earliest_divergent_command_offsets=sum(events),
                          count_with_a_measurably_different_route=len(events)))
    sums = Counter()
    for case in cases:
        sums.update(case["counts"])
    return dict(cases=cases, sums=dict(sums))


if __name__ == "__main__":
    pilot, dest = map(Path, sys.argv[1:])
    groups = {"prior_fixed_control": run(pilot.parent / "jepa_round10_fixed_five_20261124"),
              "current_v13": run(pilot)}
    project = pilot.parents[2]
    sys.path.insert(0, str(project / "jepa_env_v13/most_integration"))
    from v13_inputs import category
    accepted = project / "mapping_runtime/batch/campaigns/native_20261115/accepted/native"
    support = Counter()
    for ref in accepted.glob("*.json"):
        ep, = read(Path(read(ref)["result"]))["episodes"]
        support[category(ep["goal_text"]) or "unmapped"] += 1
    groups["description_to_prior_current_adapter_on_accepted_native_prefix"] = dict(support)
    dest.write_text(json.dumps(groups, indent=2))
