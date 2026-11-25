"""Check physical report/recovery actions, pose calls and matched terminal observations."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def load(path):
    return json.loads(path.read_bytes())


def records(path):
    if not path.is_file():
        return []
    return [json.loads(s) for s in path.read_bytes().splitlines() if s]


def locked_result(ticket):
    receipt = load(ticket)
    source = Path(receipt["result"])
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt["result_sha256"]
    doc = json.loads(raw)
    ep, = doc["episodes"]
    assert ep["episode_id"] == receipt["episode_id"]
    return doc, ep


def terminal(project):
    rt = project/"mapping_runtime"
    pilot = rt/"pilots/jepa_reid_v23_first2"
    prior = rt/"pilots/jepa_reid_v22_first2"
    rows = load(pilot/"panel.json")["rows"]
    assert rows == load(prior/"panel.json")["rows"]
    cases = []
    for row in rows:
        eid = row["episode_id"]
        started, = sorted((pilot/"comparison/attempts/second"/eid).glob("*/started.json"))
        a = rt/"runs"/started.parent.name/"agent"
        motion = {r["step"]:r for r in records(a/"action_trace.jsonl")}
        telemetry=records(a/"multiview_events.jsonl")
        ids = [r for r in telemetry if r["event"]=="pose_review"]
        events = records(a/"confirm_events.jsonl")
        reports = [r for r in events if r["event"]=="report_dispatched"]
        recovered = [r for r in events if r["event"]=="report_recovery_action"]
        decisions = records(a/"decision_trace.jsonl")
        tools = Counter(t.get("request",{}).get("name") for r in decisions for t in r.get("tools",[]))
        feedback = {r["step"] for r in decisions if r.get("world_state",{}).get("report_recovery")}
        active = Counter()
        for r in recovered:
            act = motion[r["step"]]["actual_action"]
            assert act["id"] == r["action"]
            active[act["name"]] += 1
        longest = streak = 0
        prev = -2
        for r in recovered:
            step = r["step"]
            if motion[step]["actual_action"]["name"] == "TURN_LEFT":
                streak = streak + 1 if step == prev + 1 else 1
                longest = max(longest, streak)
            else:
                streak = 0
            prev = step
        case = dict(task="clock" if row["ordinal"]==1 else "wall_art",
                    executed_actions=len(motion), report_requests_converted_to_recovery=len(recovered),
                    nonempty_live_box_steps=len({r["step"] for r in telemetry if r["event"]=="boxes" and r.get("count",0)>0 and r.get("source")=="live"}),
                    grounded_region_saves=sum(r["event"]=="object_region_saved" for r in events),
                    recovery_actions=dict(active), max_consecutive_recovery_left_turns=longest,
                    decisions_steps_exposing_recovery_feedback=len(feedback),
                    candidate_tools={k:tools[k] for k in ("propose_candidates","pick_segment","commit_candidates")},
                    pose_was_considered=len(ids), pose_actual_forward_calls=sum(bool(x.get("scored")) for x in ids),
                    pose_eligibility=dict(Counter(x.get("gate","unobserved") for x in ids)))
        ticket = pilot/"comparison/accepted/second"/(eid+".json")
        if ticket.is_file():
            original, old = locked_result(Path(row["ref"]))
            doc, ep = locked_result(ticket)
            for k in ("start_position","start_rotation","target_object_ids","target_positions","view_points","query_program"):
                assert old["paper"]["episode_definition"][k] == ep["paper"]["episode_definition"][k]
            for k in ("decision_model",):
                assert old["paper"]["agent_trace"][k] == ep["paper"]["agent_trace"][k]
            for k in ("threshold","same_floor_only","max_steps","seed","action_protocol"):
                assert original["manifest"]["arguments"][k] == doc["manifest"]["arguments"][k]
            assert len(motion) == ep["steps_taken"] and ep["max_steps_budget"]==300 and not ep["timed_out"]
            assert set(r["step"] for r in reports) == set(r["step"] for r in ep["report_events"])
            assert all(motion[r["step"]]["actual_action"]["name"]=="TARGET_FOUND" for r in reports)
            assert {i for r in ep["report_events"] for i in r["new_targets"]} == set(ep["reached_targets"])
            case.update(status="accepted", observations_registered=len(ep["paper"]["agent_trace"]["observations"]),
                instance_entries_at_end=len(ep["paper"]["agent_trace"]["final_memory"]["instances"]),
                report_details=[dict(step=r["step"],new_instances=len(r["new_targets"]),
                near_previously_found=bool(r["near_targets"] and not r["new_targets"]),
                nearest_valid_view_m=round(r["report_ne"],3) if r["report_ne"] is not None else None)
                for r in ep["report_events"]],distinct_correct=ep["found_count"],success=bool(ep["success"]),
                distinct_targets_approached=len(set(ep["approached_target_ids"])))
        else:
            assert started.with_name("quarantined.json").is_file()
            case["status"] = "engineering_interruption"
        cases.append(case)
    return cases


if __name__ == "__main__":
    print(json.dumps(terminal(Path(sys.argv[1])),allow_nan=False,indent=2))
