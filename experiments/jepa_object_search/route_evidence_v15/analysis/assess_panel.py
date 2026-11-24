"""Read only completed episodes. Reproduce physical fork actions after scoring."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys


def load(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, doc):
    encoded = (json.dumps(doc, indent=2, allow_nan=False) + "\n").encode()
    if path.exists():
        assert path.read_bytes() == encoded, "an accepted analysis already exists with different inputs"
    else:
        with path.open("xb") as f:
            f.write(encoded)


def certify_action(project, result, step, actual, other):
    from evaluation.main.evaluator import _env_action
    sys.path.insert(0, str(project / "jepa_exec_v14"))
    from verify_habitat import difference, replay_until, scene, state
    doc = load(result)
    ep, = doc["episodes"]
    timeline = ep["paper"]["steps"]
    assert timeline[step]["action"] == actual and actual != other
    env, item, data = scene(doc)
    initial = []
    final = []
    try:
        for command in (actual, other):
            replay_until(env, item, data, timeline, step)
            initial.append(state(env))
            env.step(_env_action(command))
            final.append(state(env))
    finally:
        env.close()
    pos_err, ang_err = difference(*initial)
    assert pos_err < .002 and ang_err < .03
    moved, turned = difference(initial[0], final[0])
    pos, ang = difference(*final)
    effective = moved > .005 if actual == 1 else abs(turned - 30) < .2
    return dict(verified=bool(effective and (pos > .005 or ang > 1)), moved_m=round(moved, 4),
                turned_deg=round(turned, 3), gap_m=round(pos, 4), gap_deg=round(ang, 3))


def executed_change(project, result, ep, rank, events, stop, other_index):
    group = rank["plans_native"]
    selected = group[rank["index"]]
    other = group[other_index]
    offset = next((k for k, (a, b) in enumerate(zip(selected, other)) if a != b), None)
    if offset is None:
        return None
    start = rank["step"]
    end = start + offset
    timeline = ep["paper"]["steps"]
    if end >= stop or [r["action"] for r in timeline[start:end+1]] != selected[:offset+1]:
        return dict(verified=False, reason="actual_path_did_not_reach_predicted_fork")
    if any(start < t["step"] <= end for t in ep["paper"]["agent_trace"]["decisions"]):
        return dict(verified=False, reason="intervening_host_decision")
    if rank["selected"] != rank["original"]:
        native_offset = rank["first_difference"]
        necessary = min(native_offset, offset) + 1
        dispatch = [e for e in events if e["event"] == "commit_dispatch" and
                    e.get("start") == start and e["offset"] < necessary]
        if ([(e["offset"], e["step"], e["action"]) for e in dispatch] !=
                [(k, start+k, selected[k]) for k in range(necessary)]):
            return dict(verified=False, reason="no_controller_dispatch")
        if any(e["event"] == "commit_end" and e.get("start") == start and
               e["step"] <= end+1 and e["reason"] != "first_difference_observed" for e in events):
            return dict(verified=False, reason="interrupted_commit")
    check = certify_action(project, result, end, selected[offset], other[offset])
    return dict(action_step=end, **check)


def episode(project, pilot, row, v13, diag):
    ordinal = row["ordinal"]
    eid = row["episode_id"]
    quota = row["quota"]
    ref_path = pilot / "comparison/accepted/second" / (eid + ".json")
    ref = load(ref_path)
    result = Path(ref["result"])
    ep, = load(result)["episodes"]
    replay_file = pilot / "replay" / (eid + ".json")
    outcome = diag.accepted(ref_path, replay_file, quota, False)
    native_ref = Path(row["ref"])
    assert load(native_ref)["result_sha256"] == row["result_sha256"]
    for folder in (native_ref, v13 / "comparison/accepted/second" / (eid + ".json")):
        previous = load(folder)
        baseline_path = Path(previous["result"])
        assert digest(baseline_path) == previous["result_sha256"]
        base, = load(baseline_path)["episodes"]
        assert all(base[k] == ep[k] for k in ("episode_id", "scene_id", "goal_text", "max_steps_budget",
                                              "target_mode", "required_target_count", "sensor_hfov"))
        assert base["paper"]["steps"][0]["position_before"] == ep["paper"]["steps"][0]["position_before"]
    previous_hits = min(quota, base["found_count"])
    native_hits = min(quota, load(Path(load(native_ref)["result"]))["episodes"][0]["found_count"])
    credited = set()
    surplus_step = None
    for report in ep["report_events"]:
        credited.update(report["new_targets"])
        if surplus_step is None and len(credited) > previous_hits:
            surplus_step = report["step"]
    assert len(credited) == outcome["found"]
    events = [json.loads(line) for line in (result.parent/"agent/v13_routes.jsonl").read_bytes().splitlines()]
    ranks = [r for r in events if r["event"] == "rank"]
    route_detail, confirmed = [], []
    for i, r in enumerate(ranks):
        assert r["ids"][r["index"]] == r["selected"] and r["ids"][0] == r["original"]
        assert all(math.isfinite(x) for x in r["final_scores"])
        stop = ranks[i+1]["step"] if i+1 < len(ranks) else ep["steps_taken"]
        evidence = [(idx, executed_change(project, result, ep, r, events, stop, idx))
                    for idx in sorted({0, r["v13_index"]}) if idx != r["index"]]
        checks = dict(evidence)
        diff = checks.get(r["v13_index"])
        if diff and diff["verified"]:
            confirmed.append(diff["action_step"])
        route_detail.append(dict(step=r["step"], clue_deficit=r["clue_deficit"],
                                 adaptive_weight=r["adaptive_weight"],
                                 change_from_v13=r["index"] != r["v13_index"],
                                 change_with_exploration=r["index"] != r["no_exploration_index"],
                                 native_execution=checks.get(0), v13_execution=diff))
    before_surplus = surplus_step is not None and any(t < surplus_step for t in confirmed)
    return dict(ordinal=ordinal, quota=quota, native_hits=native_hits, v13_hits=previous_hits,
                result=outcome, ranked_decisions=route_detail,
                commit_ends=dict(Counter(e["reason"] for e in events if e["event"] == "commit_end")),
                bypass_reasons=dict(Counter(e.get("reason") or "eligible" for e in events if e["event"] == "eligibility")),
                surplus_report_step=surplus_step, verified_change_before_surplus=before_surplus,
                result_sha256=digest(result), replay_sha256=digest(replay_file))


def main(project, pilot, out):
    sys.path.insert(0, str(project / "jepa_env_v13"))
    import diagnose_full_most as diag
    protocol = load(Path(__file__).with_name("protocol.json"))
    v13 = project / "mapping_runtime/pilots/jepa_round13_horizon_five"
    rows = load(pilot / "panel.json")["rows"]
    order = [r["ordinal"] for r in rows]
    assert order in (protocol["pilot_order"], protocol["conditional_order"])
    metrics = [episode(project, pilot, r, v13, diag) for r in rows]
    first = order == protocol["pilot_order"]
    data = dict(protocol_sha256=digest(Path(__file__).with_name("protocol.json")), cases=metrics)
    if first:
        assert all(c["v13_hits"] == protocol["previous_v13_capped_hits"][str(c["ordinal"])] for c in metrics)
        counts = [c["result"]["capped_found"] for c in metrics]
        condition = dict(neither_regresses=all(n >= c["v13_hits"] for c, n in zip(metrics, counts)),
                         at_least_three=sum(counts) >= 3,
                         verified_change_before_surplus=any(c["verified_change_before_surplus"] for c in metrics))
        data.update(conditions=condition, continue_phase2=all(condition.values()),
                    first_summary_sha256=digest(pilot/"five_comparison.json"))
    save(out, data)
    print(json.dumps({k:data[k] for k in ("conditions", "continue_phase2") if k in data}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("project", "pilot", "output"):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    main(a.project, a.pilot, a.output)
