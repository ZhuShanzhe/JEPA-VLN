"""Read accepted MOST episodes. Output aggregate, post-hoc scores and action receipts only."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path


def load(path):
    return json.loads(path.read_bytes())


def records(path):
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            yield from (json.loads(line) for line in stream if line.strip())


def checked(path, episode_id):
    ticket = load(path)
    source = Path(ticket["result"])
    raw = source.read_bytes()
    if sha256(raw).hexdigest() != ticket["result_sha256"]:
        raise ValueError("Accepted result hash mismatch")
    doc = json.loads(raw)
    ep, = doc["episodes"]
    if ep["episode_id"] != episode_id or ep["timed_out"] or ticket["episode_id"] != episode_id:
        raise ValueError("Receipt does not match a completed episode")
    return doc, ep, source.parent/"agent"


def audit(ticket, eid):
    doc, ep, directory = checked(ticket, eid)
    radius = float(doc["manifest"]["arguments"]["threshold"])
    paths = ep["paper"]["steps"]
    entries = {r["step"]:r for r in records(directory/"action_trace.jsonl")}
    if len(entries) != len(paths) or len(entries) != ep["steps_taken"] or not all(
            r["step"]==i and entries[i]["actual_action"]["id"]==r["action"] for i,r in enumerate(paths)):
        raise ValueError("Physical actions and action log disagree or are incomplete")
    definition = ep["paper"]["episode_definition"]
    targets = set(map(int,definition["target_object_ids"]))
    vps = definition["view_points"]
    full_viewpoints = all(vps.get(str(i)) for i in targets)
    distances = {int(k):v for k,v in ep["target_min_distances"].items()}
    if set(distances) != targets:
        raise ValueError("Incomplete official per-target distances")
    approached = {i for i,v in distances.items() if v is not None and v<=radius}
    credited = set(map(int,ep["reached_targets"]))
    if approached != set(ep["approached_target_ids"]) or not credited<=approached or len(credited)!=ep["found_count"]:
        raise ValueError("Official range or score ledger mismatch")
    actual_reports = {i for i,r in entries.items() if r["actual_action"]["name"]=="TARGET_FOUND"}
    events = ep["report_events"]
    if actual_reports != {r["step"] for r in events} or len(events)!=len(actual_reports):
        raise ValueError("Physical reports and official receipts disagree")
    by_kind = Counter()
    claimed = set()
    details = []
    for r in events:
        step = r["step"]
        near = set(map(int,r["near_targets"]))
        new = set(map(int,r["new_targets"]))
        if not new<=near or claimed & new:
            raise ValueError("Official report credit inconsistency")
        distance = r["report_ne"]
        if distance is None or not math.isfinite(distance):
            raise ValueError("No verified nearest report distance")
        if bool(near) != (distance <= radius + 1e-6):
            raise ValueError("Registered report range disagrees with post-hoc score")
        if new:
            kind = "new_target"
        elif not near:
            kind = "outside_range"
        elif r.get("over_quota_targets"):
            kind = "repeat_over_quota"
        elif r.get("within_quota_targets"):
            kind = "repeat_within_quota"
        else:
            kind = "near_without_new_credit"
        by_kind[kind]+=1
        claimed.update(new)
        details.append(dict(step=step,nearest_target_distance_m=round(distance,4), outcome=kind,
                            new_instances=len(new),over_quota_instances=len(r.get("over_quota_targets",[]))))
    if claimed!=credited:
        raise ValueError("Reported and final credited instances disagree")
    timeline = list(records(directory/"confirm_events.jsonl"))
    commands = [r for r in timeline if r["event"]=="confirm_view_action"]
    view_actions, outcomes, restore, changed = Counter(), Counter(), Counter(), Counter()
    issued = confirmed = displaced = no_displacement = not_measurable = 0
    by_step = {}
    for r in commands:
        by_step.setdefault(r["step"],[]).append(r)
        if r["step"] not in entries:
            raise ValueError("Confirmation command without physical step")
        actual = entries[r["step"]]["actual_action"]
        if r["action"] != actual["id"]:
            changed[str(r["action"])+"->"+actual["name"]]+=1
            continue
        issued += 1
        view_actions[actual["name"]]+=1
        nxt=entries.get(r["step"]+1)
        if nxt and nxt["previous_action"]["id"]==actual["id"]:
            confirmed+=1
        if actual["name"]=="MOVE_FORWARD":
            if r["step"]+1<len(paths):
                did_move=math.dist(paths[r["step"]]["position_before"],paths[r["step"]+1]["position_before"])>.001
                displaced+=int(did_move)
                no_displacement+=int(not did_move)
            else:
                not_measurable+=1
    for r in timeline:
        if r["event"]=="confirm_view_end":
            outcomes[str(r.get("reason"))]+=1
        if r["event"]=="report_recovery_action":
            restore[str(r.get("reason"))]+=1
    decisions = list(records(directory/"decision_trace.jsonl"))
    report_attempts=sum((r.get("output") or {}).get("action")=="REPORT_FOUND" for r in decisions)
    observation_rows=ep["paper"]["agent_trace"]["observations"]
    return dict(steps=len(entries),threshold_m=radius,all_targets_have_official_viewpoints=full_viewpoints,
                ground_truth_instances=len(targets),required_distinct=ep["required_target_count"],
                different_instances_ever_in_range=len(approached),distinct_correct=len(credited),
                distinct_in_range_without_credit=len(approached-credited),distinct_never_in_range=len(targets-approached),
                nearest_distance_to_each_uncredited_instance_m=sorted(round(distances[i],4) for i in targets-credited if distances[i] is not None),
                uncredited_instances_without_distance=sum(distances[i] is None for i in targets-credited),
                succeeded=bool(ep["success"]),registered_observations=len(observation_rows),
                decision_outputs_requesting_report=report_attempts,verified_physical_reports=len(events),
                outcomes=dict(by_kind),reports=details,recovery_reasons=dict(restore),
                confirm_actions_requested=len(commands),confirm_actions_equal_physical_action=issued,
                confirm_actions_equal_next_observed_previous_action=confirmed,confirm_physical_action_types=dict(view_actions),
                confirm_action_substitutions=dict(changed),steps_with_multiple_confirm_requests=sum(len(v)>1 for v in by_step.values()),
                confirm_forward_with_translation=displaced,confirm_forward_without_translation=no_displacement,
                confirm_forward_last_step_without_next_pose=not_measurable,
                confirm_closure_reasons=dict(outcomes))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("pilot",type=Path)
    parser.add_argument("--arm",choices=["second","baseline"],default="second")
    parser.add_argument("--ordinal",type=int,choices=[1,14],help="Frozen panel ordinal: 1 for clock, 14 for wall art")
    args=parser.parse_args()
    result={}
    for row in load(args.pilot/"panel.json")["rows"]:
        if args.ordinal is not None and row["ordinal"]!=args.ordinal:
            continue
        eid=row["episode_id"]
        ticket=Path(row["ref"]) if args.arm=="baseline" else args.pilot/"comparison/accepted/second"/(eid+".json")
        result["clock" if row["ordinal"]==1 else "wall_art"]=audit(ticket,eid)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
