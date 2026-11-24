"""Replay candidate admission at every archived native exploration decision."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys


def read_json(path):
    return json.loads(path.read_bytes())


def events(path, name):
    return [row for line in path.read_bytes().splitlines() if line
            and (row := json.loads(line))["event"] == name]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_episode(pilot, row, build_table):
    episode_id = row["episode_id"]
    ref = read_json(pilot / "comparison/accepted/second" / (episode_id + ".json"))
    result = Path(ref["result"])
    if digest(result) != ref["result_sha256"]:
        raise ValueError("Result does not match accepted receipt")
    episode, = read_json(result)["episodes"]
    limit = int(episode["max_steps_budget"])
    agent = result.parent / "agent"
    probes = events(agent / "frontier_probe.jsonl", "activation")
    grouped = defaultdict(list)
    for probe in probes:
        if probe["activated"]:
            grouped[probe["step"], probe["source"], probe["selected_id"]].append(probe)
    checks = events(agent / "v13_routes.jsonl", "eligibility")
    old_count = new_count = retained = recovered = lost = 0
    fallback, recovered_from, route_counts, lengths = Counter(), Counter(), Counter(), Counter()
    native_counts = Counter()
    first_old = first_new = None
    guards = {"unsupported_target_category", "native_action_not_explore", "off_level"}
    for check in checks:
        step, source, original = check["step"], check["source"], check["original"]
        probelist = grouped[(step, source, original)]
        if len(probelist) != 1:
            raise ValueError("Cannot uniquely match native activation with eligibility record")
        rows = probelist.pop()["candidates"]
        if len(rows) != check["native_candidate_count"]:
            raise ValueError("Logged candidate counts differ")
        native_counts[len(rows)] += 1
        old = check["reason"]
        if old is None:
            old_count += 1
            if first_old is None:
                first_old = step
        if old in guards:
            entries, reason = [], old
        else:
            entries, reason = build_table(rows, original, limit - step - 4)
        if reason:
            fallback[reason] += 1
            lost += int(old is None)
            continue
        plans = [tuple(e["plan"]) for e in entries]
        assert entries[0]["id"] == original and len(plans) >= 2
        assert all(1 in p and len(p) <= min(24, limit-step-4) for p in plans)
        for i, left in enumerate(plans):
            for right in plans[i+1:]:
                assert any(a != b for a,b in zip(left, right)), "no executable split"
        new_count += 1
        retained += int(old is None)
        if old is not None:
            recovered_from[old] += 1
            recovered += 1
        route_counts[len(entries)] += 1
        lengths.update(len(p) for p in plans)
        if first_new is None:
            first_new = step
    assert len(checks) == sum(fallback.values()) + new_count
    assert old_count == lost + retained and new_count == recovered + retained
    return dict(ordinal=row["ordinal"], checks=len(checks), old_eligible=old_count,
                new_eligible=new_count, retained=retained, newly_eligible=recovered,
                previously_eligible_now_rejected=lost, first_old_step=first_old, first_new_step=first_new,
                raw_candidate_counts=dict(native_counts), accepted_route_counts=dict(route_counts),
                route_lengths=dict(sorted(lengths.items())), fallback=dict(fallback),
                newly_eligible_by_previous_reason=dict(recovered_from))


def summarize(cases):
    sums = ("checks", "old_eligible", "new_eligible", "newly_eligible", "retained",
            "previously_eligible_now_rejected")
    hist = ("raw_candidate_counts", "accepted_route_counts", "route_lengths", "fallback",
            "newly_eligible_by_previous_reason")
    result = {key:sum(c[key] for c in cases) for key in sums}
    for key in hist:
        aggregate = Counter()
        for case in cases:
            aggregate.update(case[key])
        result[key] = dict(sorted(aggregate.items()))
    return result


def main(project, out):
    adapter = project / "jepa_variable_v16/adapter"
    sys.path.insert(0, str(adapter))
    from variable_routes import build_table
    benchmarks = {}
    for version, directory, order in [
        ("v15_primary", "jepa_evidence_v15_first2", [1,14]),
        ("v13_supplementary", "jepa_round13_horizon_five", [1,14,21,32,39]),
    ]:
        pilot = project / "mapping_runtime/pilots" / directory
        rows = read_json(pilot / "panel.json")["rows"]
        assert [r["ordinal"] for r in rows] == order
        cases = [audit_episode(pilot, row, build_table) for row in rows]
        benchmarks[version] = dict(cases=cases, total=summarize(cases))
    doc = dict(adapter_input_sha256=digest(adapter / "variable_routes.py"),
               scope="Eligibility on all recorded native activations of accepted historical trajectories. "
                     "Does not rerun the policy, verify scan completion, or estimate episode success. "
                     "The two panels share task IDs and must be reported separately.",
               panels=benchmarks)
    out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, panel in benchmarks.items():
        print(name, json.dumps(panel["total"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.project, args.output)
