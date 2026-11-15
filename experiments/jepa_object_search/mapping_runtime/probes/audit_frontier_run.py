"""Post-episode verifier. The agent never imports this module or its judge data."""
import argparse
from collections import Counter
import json
from pathlib import Path


def audit(episode, probe):
    eid = episode["episode_id"]
    rows = [r for r in probe if r["episode"] == eid]
    actions = [r for r in rows if r["event"] == "returned_action"]
    steps = episode["paper"]["steps"]
    decisions = episode["paper"]["agent_trace"]["decisions"]
    if [(r["step"], r["action_id"]) for r in actions] != [(r["step"], r["action"]) for r in steps]:
        raise ValueError("Agent return and evaluator action sequences do not match for " + eid)
    if [r["step"] for r in steps if r["action_role"] == "target_found"] != [
            r["step"] for r in episode["report_events"]]:
        raise ValueError("Report actions do not match official report event steps")
    if not steps:
        raise ValueError("No executed evaluator steps in " + eid)
    selections = []
    for row in rows:
        if row["event"] != "activation" or not row["activated"]:
            continue
        valid = {r["id"] for r in row["candidates"]}
        if row["selected_id"] not in valid:
            raise ValueError("Activation outside valid author table")
        matches = [d for d in decisions if d.get("step") == row["step"]
                   and d.get("decision_origin") == "vlm" and d.get("validation") == "ok"
                   and (d.get("output") or {}).get("action") == "GOTO_FRONTIER"
                   and d["output"].get("target_id") == row["selected_id"]
                   and row["selected_id"] in {f["id"] for f in d["world_state"]["frontiers"]}]
        selections.append({"step": row["step"], "revision": row["revision"],
                           "id": row["selected_id"], "legal_count": len(valid),
                           "source": row["source"],
                           "accepted_model_choice": row["source"] == "decider GOTO_FRONTIER " + row["selected_id"]
                           and bool(matches)})
    inputs = [r for r in rows if r["event"] == "decision_input"]
    return {"episode": eid, "actions_checked": len(steps),
            "pose_free": actions[0]["pose_free"], "scale_source": actions[0]["scale_source"],
            "policy_returns_and_evaluator_actions_match": True,
            "agent_report_actions_and_judge_events_match": True,
            "author_decision_origins": dict(Counter(d.get("decision_origin") for d in decisions)),
            "accepted_model_actions": dict(Counter(d["output"]["action"] for d in decisions
                                                    if d.get("decision_origin") == "vlm")),
            "max_candidates": max((len(r["candidates"]) for r in inputs), default=0),
            "empty_table_observations": sum(not r["candidates"] for r in inputs),
            "activations": selections, "reports": len(episode["report_events"]),
            "official_tp": episode["true_positive_count"], "official_fp": episode["false_positive_count"],
            "found_count": episode["found_count"], "end_reason": episode["failure_reason"],
            "finished_by_agent": episode["finished_by_agent"], "timed_out": episode["timed_out"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("official_result", type=Path)
    parser.add_argument("frontier_probe", type=Path)
    args = parser.parse_args()
    result = json.loads(args.official_result.read_text())
    trace = [json.loads(line) for line in args.frontier_probe.read_text().splitlines() if line]
    print(json.dumps([audit(e, trace) for e in result["episodes"]], indent=2))
