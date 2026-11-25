"""Aggregate an accepted MOST-Nav cohort and matched independent semantic replays."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def aggregate(project):
    runtime = project / "mapping_runtime"
    campaign_name = "native_20261115"
    campaign = runtime / "batch/campaigns" / campaign_name
    replay_dir = runtime / "analysis/visibility" / campaign_name / "verified"
    receipts = sorted((campaign / "accepted/native").glob("*.json"))
    diagnosis = read(campaign / "analysis/accepted_diagnosis.json")
    assert receipts and len(receipts) == diagnosis["accepted"]
    budget, window = 300, 50
    counts, tail = Counter(), Counter()
    for path in receipts:
        receipt = read(path)
        raw = Path(receipt["result"]).read_bytes()
        replay = read(replay_dir / path.name)
        assert hashlib.sha256(raw).hexdigest() == receipt["result_sha256"] == replay["source_result_sha256"]
        episode, = json.loads(raw)["episodes"]
        quota = episode["required_target_count"]
        steps = episode["steps_taken"]
        assert episode["max_steps_budget"] == budget and 0 < steps <= budget
        reached = set(episode["approached_target_ids"])
        found = set(episode["reached_targets"])
        assert found <= reached and bool(episode["sr"]) == (len(found) >= quota)
        if len(found) >= quota:
            counts["success"] += 1
            continue
        hit_cap = steps == budget
        enough = len(reached) >= quota
        reason = "cap" if hit_cap else "early"
        coverage = "enough_in_range" if enough else "insufficient_in_range"
        counts[f"{reason}_{coverage}"] += 1
        if not hit_cap:
            continue
        reports = episode["report_events"]
        tail["episodes"] += 1
        tail["no_report_in_entire_episode"] += not reports
        recent_credits = sum(len(report["new_targets"]) for report in reports if report["step"] >= budget - window)
        first_clear = [target["exposure"]["moderate"]["first_qualifying_step"] for target in replay["targets"]]
        recent_exposures = sum(step is not None and budget - window <= step < budget for step in first_clear)
        tail["episodes_with_new_credit"] += bool(recent_credits)
        tail["episodes_with_first_clear_target_exposure"] += bool(recent_exposures)
        tail["episodes_with_either"] += bool(recent_credits or recent_exposures)
        tail["episodes_with_neither"] += not (recent_credits or recent_exposures)
        tail["new_instance_credits"] += recent_credits
        tail["first_clear_target_exposures"] += recent_exposures

    groups = diagnosis["groups"]["all"]
    parts = groups["failure_partition"]
    assert parts["diagnostic_unknown"] == 0
    assert parts["success"] == counts["success"]
    assert parts["insufficient_approach_quota"] == counts["early_insufficient_in_range"] + counts["cap_insufficient_in_range"]
    assert parts["sufficient_approach_but_underreported"] == counts["early_enough_in_range"] + counts["cap_enough_in_range"]
    assert tail["episodes"] == groups["termination"]["max_steps"] == tail["episodes_with_either"] + tail["episodes_with_neither"]
    return dict(planned_episodes=diagnosis["expected"], accepted_episodes=len(receipts), step_limit=budget,
                terminal_outcomes=dict(counts), task_quota_shortfall=groups["quota_shortfall"],
                last_window_zero_based_inclusive=[budget-window, budget-1],
                capped_failed_last_window=dict(tail))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    print(json.dumps(aggregate(parser.parse_args().project_root), indent=2, sort_keys=True))
