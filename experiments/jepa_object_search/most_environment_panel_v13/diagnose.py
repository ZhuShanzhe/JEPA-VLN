"""Export numeric, source-verified MOST-Nav case diagnostics. Never launches a model."""
import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path


ORDINAL_QUOTA = {1: 2, 14: 3, 21: 3, 32: 3, 39: 5}


def read(path):
    return json.loads(path.read_bytes())


def require(ok, tag):
    if not ok:
        raise ValueError(tag)


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def motion(ep):
    rows = ep["paper"]["steps"]
    positions = [r["position_before"] for r in rows]
    positions.append(ep["trajectory_positions"][-1]["position"])
    forward = stopped = rotate = still = longest = streak = 0
    travelled = 0.
    for i, (row, before, after) in enumerate(zip(rows, positions, positions[1:])):
        require(row["step"] == i, "step_index_mismatch")
        distance = math.dist(before, after)
        require(math.isfinite(distance), "bad_position_distance")
        travelled += distance
        is_forward = row["action_role"] == "navigate" and row["action"] == 1
        forward += is_forward
        rotate += row["action_role"] == "navigate" and row["action"] in (2, 3)
        failed_forward = is_forward and distance < .01
        stopped += failed_forward
        still += distance < .01
        streak = streak + 1 if failed_forward else 0
        longest = max(longest, streak)
    require(abs(travelled - ep["trajectory_length"]) <= .05, "pose_length_differs")
    return dict(forward_commands=forward, turns=rotate, near_stationary_forward=stopped,
                longest_consecutive_near_stationary_forward=longest,
                all_commands_under_1cm=still, path_metres=round(travelled, 5))


def sensor_and_reports(ep, replay, quota):
    hits = set(map(int, ep["reached_targets"]))
    near = set(map(int, ep["approached_target_ids"]))
    targets = replay["targets"]
    require(replay["thresholds"]["moderate"] == [103, 2] and replay["semantic_resolution"] == [320, 320],
            "visibility_protocol_changed")
    exposed = {int(x["id"]) for x in targets if x["exposure"]["moderate"]["first_qualifying_step"] is not None}
    require({int(x["id"]) for x in targets if x["reached"]} == hits, "replay_report_mismatch")
    require({int(x["id"]) for x in targets if x["approached"]} == near, "replay_approach_mismatch")
    require(hits <= near and len(hits) == ep["found_count"] and
            len(exposed) == replay["count_exposed"]["moderate"], "replay_counts_disagree")
    reports = ep["report_events"]
    claimed = set()
    repeated = empty = new = 0
    for event in reports:
        newly = set(map(int, event["new_targets"]))
        in_range = set(map(int, event["near_targets"]))
        require(newly == in_range - claimed and in_range <= near, "report_sequence_mismatch")
        new += bool(newly)
        repeated += bool(in_range) and not newly
        empty += not in_range
        claimed.update(newly)
    require(claimed == hits and new + repeated + empty == len(reports), "report_count_mismatch")
    found = len(hits)
    approached = len(near)
    success = found >= quota
    require(bool(ep["success"]) == success, "success_rule_mismatch")
    return dict(quota=quota, found=found, capped_found=min(found, quota), success=success,
                official_recall=ep["recall"], official_precision=ep["precision"],
                official_spl_multi=ep["spl_multi"], approached=approached, clear_exposure=len(exposed),
                clear_and_approached=len(exposed & near), clear_never_approached=len(exposed-near),
                clear_unreported=len(exposed-hits), approached_unreported=approached-found,
                approached_and_clear_unreported=len((near & exposed)-hits),
                required_missing_never_approached=max(0, quota-approached),
                required_missing_approached_unreported=min(approached, quota)-min(found, quota),
                reports=len(reports), reports_yielding_new_instances=new,
                repeat_only_reports=repeated, no_target_in_range_reports=empty,
                credited_with_no_clear_exposure=len(hits-exposed),
                termination="quota_met" if success else
                            "step_budget" if len(ep["paper"]["steps"]) >= ep["max_steps_budget"] else
                            "agent_finished_under_quota" if ep.get("finished_by_agent") else "other",
                steps=len(ep["paper"]["steps"]))


def interventions(folder, ep):
    f = folder / "agent/v13_routes.jsonl"
    if not f.is_file():
        raise ValueError("missing_route_diagnostic")
    rows = [json.loads(x) for x in f.read_bytes().splitlines() if x.strip()]
    ranks = [r for r in rows if r["event"] == "rank"]
    opening = [r for r in rows if r["event"] == "episode"]
    require(len(opening) == 1 and opening[0]["step"] == 0, "invalid_route_trace_episode")
    eligibility_reasons = Counter()
    known_reasons = {"native_action_not_explore", "off_level", "remaining_budget_below_28",
                     "unsupported_target_category", "native_plan_short", "fewer_than_three_distinct_native_plans"}
    for r in rows:
        if r["event"] == "eligibility" and r["reason"]:
            reason = r["reason"] if r["reason"] in known_reasons else "other"
            eligibility_reasons[reason] += 1
    scans = [r for r in rows if r["event"] == "scan_end"]
    score_disagreements = Counter()
    for rank in ranks:
        cue_steps = rank["frame_steps"]
        require(len(cue_steps) == 3 and all(t < rank["step"] for t in cue_steps) and
                sorted(set(cue_steps)) == cue_steps, "future_or_missing_scan")
        ids = rank["ids"]
        require(ids[0] == rank["original"] and len(ids) == len(set(ids)) == 3 and
                ids[rank["index"]] == rank["selected"], "candidate_identity_mismatch")
        for arm in ("v11", "direct", "jepa"):
            scores = rank[arm+"_scores"]
            require(len(scores) == 3 and all(map(math.isfinite, scores)), "rank_logits_invalid")
            score_disagreements[arm+"_raw_head_argmax_differs_from_final"] += (
                ids[max(range(3), key=lambda j: scores[j])] != rank["selected"])
    disagreement = sum(r["selected"] != r["original"] for r in ranks)
    count = Counter(r["event"] for r in rows)
    return dict(status="recorded", events=len(rows), eligible_checks=count["eligibility"],
                room_prior_supported=bool(opening[0]["room_prior_supported"]),
                bypass_events=sum(eligibility_reasons.values()), bypass_reasons=dict(eligibility_reasons),
                scan_started=count["scan_start"], scan_completed=sum(r["reason"]=="complete" for r in scans),
                scans_ended_early=sum(r["reason"]!="complete" for r in scans),
                charged_scan_turns_with_return_observation=sum(r["physical_turns"] for r in scans),
                scans_still_open_at_termination=count["scan_start"]-len(scans),
                ranked_decisions=len(ranks), shadow_raw_comparisons=dict(score_disagreements),
                non_native_route_selected=disagreement,
                **verified_forks(rows, ranks, ep))


def verified_forks(events, ranks, ep):
    """Count a fork once, only after its first different command appears in the accepted trajectory."""
    rows = ep["paper"]["steps"]
    actions = [s["action"] for s in rows]
    decisions = [r["step"] for r in ep["paper"]["agent_trace"]["decisions"]]
    fork_steps = []
    evidenced = no_evidence = 0
    for idx, rank in enumerate(ranks):
        if rank["original"] == rank["selected"]:
            continue
        plans = rank["plans_native"]
        if isinstance(plans, list):
            plans = dict(zip(rank["ids"], plans))
        a, b = plans[rank["original"]], plans[rank["selected"]]
        first = next((j for j, (orig, selected) in enumerate(zip(a, b)) if orig != selected), None)
        require(first == rank["first_difference"], "planned_fork_disagrees")
        if first is None:
            no_evidence += 1
            continue
        # New rank begins only after the three paid-for physical side views and heading return.
        start = rank.get("step")
        if not isinstance(start, int):
            no_evidence += 1
            continue
        end = ranks[idx+1].get("step", len(actions)) if idx+1 < len(ranks) else len(actions)
        confirmations = [e for e in events if e["event"] == "route_progress"
                         and e.get("start") == start and e.get("offset") == first
                         and e["step"] == start+first+1 and e.get("action") == b[first]
                         and e.get("divergence_reached") is True]
        interrupted = any(start < s <= start+first for s in decisions)
        if confirmations and start + first < end and not interrupted and actions[start:start+first+1] == b[:first+1]:
            evidenced += 1
            fork_steps.append(start+first)
        else:
            no_evidence += 1
    new_before = new_after = None
    if fork_steps:
        first = min(fork_steps)
        new_before = sum(len(r["new_targets"]) for r in ep["report_events"] if r["step"] <= first)
        new_after = sum(len(r["new_targets"]) for r in ep["report_events"] if r["step"] > first)
    return dict(verified_first_divergent_commands=evidenced, selected_without_verified_fork=no_evidence,
                first_verified_fork_step=min(fork_steps) if fork_steps else None,
                credited_before_or_at_first_fork=new_before, credited_after_first_fork=new_after)


def accepted(reference, replay_path, quota, trace):
    ref = read(reference)
    result = Path(ref["result"])
    require(file_sha(result) == ref["result_sha256"], "result_sha_mismatch")
    require(replay_path.is_file(), "independent_replay_pending")
    replay = read(replay_path)
    require(replay["source_result_sha256"] == ref["result_sha256"], "independent_replay_sha_mismatch")
    ep, = read(result)["episodes"]
    require(ep["episode_id"] == replay["episode_id"] == ref["episode_id"], "identity_mismatch")
    require(ep["required_target_count"] == quota and ep["max_steps_budget"] == 300, "protocol_mismatch")
    require(ep["steps_taken"] == replay["steps_replayed"] == len(ep["paper"]["steps"]) and
            replay["max_position_error_m"] <= .002 and replay["reference_rgb_frames"] > 0 and
            replay["max_reference_rgb_mae_at_64"] < 3.0 and
            ep.get("failure_reason") in (None, "incomplete_finish", "max_steps") and
            not ep.get("timed_out"), "replay_or_engineering_failure")
    row = sensor_and_reports(ep, replay, quota)
    row["motion"] = motion(ep)
    if trace:
        row["intervention"] = interventions(result.parent, ep)
    return row


def quarantines(pilot, eid):
    folders = set()
    for area in (pilot/"comparison/attempts/second"/eid, pilot/"retry_archive"):
        if area.is_dir():
            for q in area.rglob("quarantined.json"):
                if area == pilot/"retry_archive" and eid not in q.parts:
                    continue
                require(not read(q).get("score_included", False), "quarantine_is_scored")
                folders.add(q.parent.name)
    return len(folders)


def summarize(rows, mode):
    data = [r[mode] for r in rows if isinstance(r.get(mode), dict)]
    if not data:
        return {"accepted_episodes":0}
    sum_keys = ("capped_found", "quota", "found", "success", "approached", "clear_exposure",
                "approached_unreported", "clear_unreported", "required_missing_never_approached",
                "required_missing_approached_unreported", "reports", "repeat_only_reports", "no_target_in_range_reports")
    return dict(accepted_episodes=len(data), sums={k:sum(r[k] for r in data) for k in sum_keys},
                mean_episode_recall=statistics.mean(r["official_recall"] for r in data),
                mean_official_precision=statistics.mean(r["official_precision"] for r in data),
                mean_official_spl_multi=statistics.mean(r["official_spl_multi"] for r in data))


def run(pilot, native_replays, trace=True):
    panel = read(pilot/"panel.json")["rows"]
    require([(r["ordinal"],r["quota"]) for r in panel] == list(ORDINAL_QUOTA.items()), "wrong_panel")
    output = []
    for item in panel:
        eid, quota = item["episode_id"], item["quota"]
        native = Path(item["ref"])
        require(read(native)["result_sha256"] == item["result_sha256"], "original_panel_sha_mismatch")
        row = dict(ordinal=item["ordinal"], native=accepted(native, native_replays/(eid+".json"), quota, False))
        ref = pilot/"comparison/accepted/second"/(eid+".json")
        replay = pilot/"replay"/(eid+".json")
        row["quarantined_attempts"] = quarantines(pilot, eid)
        if not ref.is_file():
            row["status"] = "no_accepted_result"
        elif not replay.is_file():
            row["status"] = "awaiting_physical_replay"
        else:
            row["tested"] = accepted(ref, replay, quota, trace)
            row["status"] = "accepted_replayed"
            row["capped_found_difference"] = row["tested"]["capped_found"] - row["native"]["capped_found"]
        output.append(row)
    pairs = [r for r in output if "tested" in r]
    return dict(model="gpt-4o-2024-11-20", panel_sha256=file_sha(pilot/"panel.json"),
                step_budget=300, cases=output, panel_status=dict(Counter(r["status"] for r in output)),
                native_all=summarize(output, "native"), native_paired=summarize(pairs,"native"),
                tested_paired=summarize(pairs, "tested"), quarantined_attempts=sum(r["quarantined_attempts"] for r in output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("--native-replays", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--historical-control", action="store_true", help="read an older intervention panel without V13 logging")
    args = parser.parse_args()
    pilot = args.pilot.resolve()
    default = pilot.parents[1]/"analysis/visibility/native_20261115/verified"
    summary = run(pilot, args.native_replays or default, not args.historical_control)
    serialized = json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False)+"\n"
    if args.output is None:
        print(serialized, end="")
    else:
        tmp = args.output.with_name(args.output.name + "." + str(os.getpid()) + ".tmp")
        with tmp.open("x") as f:
            f.write(serialized)
        tmp.replace(args.output)
        print(json.dumps(summary["panel_status"]))


if __name__ == "__main__":
    main()
