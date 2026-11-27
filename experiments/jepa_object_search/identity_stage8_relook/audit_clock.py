"""Inspect one accepted, paired MOST-Nav development episode without exposing its raw observations."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


PROJECT = Path(__file__).resolve().parent.parent
STUDIES = PROJECT / "mapping_runtime/pilots"
STAGE8 = STUDIES / "identity_stage8_weak_relook"
STAGE7 = STUDIES / "identity_stage7_arrival_transfer"


def load(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lines(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def accepted(ticket_path):
    ticket = load(ticket_path)
    path = Path(ticket["result"])
    assert digest(path) == ticket["result_sha256"]
    doc = load(path)
    ep, = doc["episodes"]
    actions = lines(path.parent / "agent/action_trace.jsonl")
    steps = ep["paper"]["steps"]
    assert len(steps) == ep["steps_taken"] == len(actions) <= ep["max_steps_budget"] == 300
    assert not ep["timed_out"] and ep["required_target_count"] == 2
    assert all(i == a["step"] == s["step"] and a["actual_action"]["id"] == s["action"]
               for i, (a, s) in enumerate(zip(actions, steps)))
    events = ep["report_events"]
    assert [e["step"] for e in events] == [a["step"] for a in actions if a["actual_action"]["name"] == "TARGET_FOUND"]
    credited = set()
    for e in events:
        new = set(e["new_targets"])
        assert new.issubset(e["near_targets"]) and not (credited & new)
        credited.update(new)
    assert credited == set(ep["reached_targets"]) and len(credited) == ep["found_count"]
    assert ep["success"] == (len(credited) >= ep["required_target_count"])
    return path, doc, ep


def score(ep):
    rr = ep["report_events"]
    return dict(actions=ep["steps_taken"], budget=ep["max_steps_budget"], quota=ep["required_target_count"],
                correct_distinct=ep["found_count"], complete=bool(ep["success"]),
                report_actions=len(rr), repeats=ep["repeat_report_count"],
                out_of_range=sum(not e["near_targets"] for e in rr),
                official_false_positive_count=ep["false_positive_count"],
                distinct_in_range=len(set(ep["approached_target_ids"])),
                sorted_minimum_report_distances_m=sorted(round(v, 3) for v in ep["target_min_distances"].values()
                                                         if isinstance(v, (int,float)) and math.isfinite(v)),
                stopped_by_agent=bool(ep["finished_by_agent"]),
                credit_events=[dict(step=e["step"],new=len(e["new_targets"]),in_range=bool(e["near_targets"]))
                               for e in rr])


def check_pair(doc, other):
    e, = doc["episodes"]
    ref, = other["episodes"]
    for key in ("episode_id","scene_id","object_category","goal_text","target_mode","required_target_count",
                "sensor_width","sensor_height","sensor_hfov","max_steps_budget","max_steps_per_target","action_protocol"):
        assert e[key] == ref[key], key
    for key in ("seed","threshold","max_steps","same_floor_only","pool_match_threshold"):
        assert doc["manifest"]["arguments"][key] == other["manifest"]["arguments"][key], key
    assert e["paper"]["agent_trace"]["decision_model"] == ref["paper"]["agent_trace"]["decision_model"]
    assert e["paper"]["steps"][0]["position_before"] == ref["paper"]["steps"][0]["position_before"]


def visibility(doc, weak_steps):
    """Replay real actions and score only after the policy has completed. Never input the labels to the agent."""
    import numpy as np
    from check_routes import semantic_scene
    from evaluation.main.evaluator import _env_action, _near_target_ids, _reset_environment

    ep, = doc["episodes"]
    env, item, data = semantic_scene(doc)
    threshold = doc["manifest"]["arguments"]["threshold"]
    report = {r["step"]:r for r in ep["report_events"]}
    targets = set(map(int, item.target_object_ids))
    metric = defaultdict(Counter)
    seen, reached, credited = set(), set(), set()
    def near_position(pos):
        return set(map(int, _near_target_ids(pos, item.target_object_ids, item.target_positions,
                                            threshold=threshold,pathfinder=env.sim.pathfinder,target_view_points=item.view_points)))
    largest_error = 0.
    try:
        obs = _reset_environment(env,item,data)
        for step in ep["paper"]["steps"]:
            i = step["step"]
            pos = env.sim.get_agent_state().position
            error = float(np.linalg.norm(np.asarray(pos)-step["position_before"]))
            largest_error = max(largest_error,error)
            assert error < .002
            mask = np.asarray(obs["semantic"]).squeeze()
            assert mask.shape == (320,320)
            clear = {t for t in targets if np.count_nonzero(mask == t) >= 103}
            near = near_position(pos)
            reached.update(near)
            windows = ["all"]
            if i-1 in weak_steps:
                windows.append("observation_after_executed_weak_cue_action")
            for window in windows:
                counter = metric[window]
                counter["steps"] += 1
                counter["uncredited_in_view"] += bool(clear-credited)
                counter["uncredited_in_range"] += bool(near-credited)
                counter["uncredited_in_range_and_view"] += bool((near & clear)-credited)
                counter["previously_unseen_target_first_visible"] += len(clear-seen)
            seen.update(clear)
            if i in report:
                assert near == set(map(int,report[i]["near_targets"]))
                credited.update(map(int,report[i]["new_targets"]))
            if step["action_role"] == "navigate":
                obs = env.step(_env_action(step["action"]))
        reached.update(near_position(env.sim.get_agent_state().position))
        assert reached == set(map(int, ep["approached_target_ids"]))
    finally:
        env.close()
    return dict(clear_threshold="at least 103 target-labeled pixels in a 320x320 semantic frame",
                distinct_ever_visible=len(seen), distinct_ever_in_range=len(reached),
                max_replay_position_error_m=largest_error, frame_counts=dict(metric))


def safe_code(value):
    return value if isinstance(value,str) and re.fullmatch(r"[a-z][a-z0-9_]{0,79}",value) else "other"


def route_evidence(rows, ep):
    steps = ep["paper"]["steps"]
    moves = defaultdict(set)
    events = Counter()
    outcomes = defaultdict(Counter)
    for r in rows:
        name = safe_code(r["event"])
        events[name] += 1
        if name=="weak_relook_action":
            moves[int(r["step"])].add(int(r["action"]))
        if r.get("reason"):
            outcomes[name][safe_code(r["reason"])] += 1
    executed = {t for t,proposals in moves.items() if 0<=t<len(steps) and steps[t]["action"] in proposals}
    translated = []
    for t in sorted(executed):
        if steps[t]["action"]!=1 or t+1>=len(steps):
            continue
        before,after=steps[t]["position_before"],steps[t+1]["position_before"]
        translated.append(math.hypot(after[0]-before[0],after[2]-before[2]))
    departures=[]
    for r in rows:
        if r["event"]!="weak_region_departure":
            continue
        t=int(r["step"])
        if not 0<=t<len(steps):
            continue
        origin=steps[t]["position_before"]
        positions=[s["position_before"] for s in steps[t+1:t+17]]
        departures.append(dict(step=t,distinct_frontier=bool(r.get("distinct_frontier")),
                               observed_following_actions=len(positions),
                               max_horizontal_displacement_next_16_m=round(max(
                                   (math.hypot(p[0]-origin[0],p[2]-origin[2]) for p in positions),default=0.),3)))
    stats=dict(events=dict(events),reasons=dict(outcomes),distinct_action_proposal_steps=len(moves),
               proposals_matching_executed_action=len(executed),forward_actions_with_following_pose=len(translated),
               forwards_over_5cm=sum(d>.05 for d in translated),departure_followups=departures)
    return stats,executed


def audit(study, reference_study, include_physics=False):
    row, = load(study/"panel.json")["rows"]
    name = row["episode_id"]+".json"
    assert (study/"closed.json").exists(), "Missing campaign closure"
    assert digest(study/"panel.json") == load(study/"launch_receipt.json")["panel_sha256"]
    native_path, native_doc, native = accepted(row["ref"])
    assert row["result_sha256"] == digest(native_path)
    path, doc, ep = accepted(study/"comparison/accepted/second"/name)
    assert digest(path) == load(study/"score_receipts"/name)["source_result_sha256"]
    check_pair(doc, native_doc)
    previous = PROJECT/"identity_stage7_transfer/clock_adapted_canonical.json"
    replay = load(previous)
    assert replay["mode"] == "adapted"
    assert digest(native_path) == load(PROJECT/"identity_stage7_transfer/clock_native_canonical.json")["source_sha256"]
    reference_row, = load(reference_study/"panel.json")["rows"]
    assert row == reference_row
    previous_ticket = reference_study/"comparison/accepted/second"/name
    decisions = lines(path.parent/"agent/decision_trace.jsonl")
    confirms = lines(path.parent/"agent/confirm_events.jsonl")
    weak = [r for r in confirms if r["event"].startswith("weak_")]
    route,weak_steps = route_evidence(weak,ep)
    refs = [r for r in confirms if r["event"]=="reference_quality"]
    deferrals = Counter(r["reason"] for r in confirms if r["event"]=="report_deferred")
    receipts = Counter(r["event"] for r in confirms if r["event"] in {
        "report_dispatched","report_action_acknowledged","report_owner_unresolved","report_not_acknowledged",
        "report_already_recorded","confirmed_reference_pinned"})
    report_steps = {e["step"] for e in ep["report_events"]}
    dispatched = {int(r["step"]) for r in confirms if r["event"]=="report_dispatched"}
    acked = {int(r["source_step"]) for r in confirms if r["event"]=="report_action_acknowledged"}
    prev_metrics = dict(kind="fixed_motion_replay",correct_distinct=replay["credited_distinct"],
                        quota=replay["quota"], reports=replay["reports"], repeats=replay["repeat_reports"],
                        out_of_range=replay["out_of_range_reports"], original_result_sha256=replay["original_trajectory_sha256"],
                        actual_gpt_requests=len(replay["requests"]))
    if previous_ticket.exists():
        pp, pd, pe = accepted(previous_ticket)
        check_pair(doc,pd)
        prev_metrics = dict(kind="accepted_autonomous",result_sha256=digest(pp),**score(pe))
    source = {p.name:digest(p) for p in [path, path.parent/"agent/decision_trace.jsonl",
              path.parent/"agent/confirm_events.jsonl",path.parent/"agent/action_trace.jsonl"]}
    output = dict(original_most_nav=dict(kind="accepted_autonomous",result_sha256=digest(native_path),**score(native)),
                  previous_stage7=prev_metrics,
                  evaluated_policy=dict(kind="accepted_autonomous",entry=doc["manifest"]["arguments"]["agent"],**score(ep)),
                  paired_protocol=dict(seed=doc["manifest"]["arguments"]["seed"],
                                       report_radius_m=doc["manifest"]["arguments"]["threshold"],
                                       actions=ep["action_protocol"],budget=ep["max_steps_budget"],
                                       model=ep["paper"]["agent_trace"]["decision_model"]),
                  source_hashes=source,
                  evidence=dict(weak_cue_route=route,receipt_events=dict(receipts),
                                executed_report_steps_with_dispatch=len(report_steps & dispatched),
                                executed_report_steps_with_memory_receipt=len(report_steps & acked),
                                trusted_reference_checks=sum(bool(r["trusted"]) for r in refs),
                                all_reference_checks=len(refs),reference_outcomes=dict(Counter(r["reason"] for r in refs)),
                                report_deferrals=dict(deferrals),
                                new_vlm_report_requests=sum(d["decision_origin"]=="vlm" and
                                    (d.get("output") or {}).get("action")=="REPORT_FOUND" for d in decisions)))
    if include_physics:
        output["offline_visibility"] = visibility(doc,weak_steps)
    return output


if __name__ == "__main__":
    from verify_rank import deny_ip_sockets
    deny_ip_sockets()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,default=STAGE8)
    p.add_argument("--output",type=Path)
    p.add_argument("--with-relook-visibility",action="store_true",
                   help="Replay actions in Habitat to measure clear views after executed route actions")
    args = p.parse_args()
    result = audit(args.study,STAGE7,args.with_relook_visibility)
    text = json.dumps(result,indent=2,allow_nan=False)+"\n"
    if args.output:
        if args.output.exists() and args.output.read_text()!=text:
            raise ValueError("Audit destination contains a different result")
        args.output.write_text(text)
    print(text)
