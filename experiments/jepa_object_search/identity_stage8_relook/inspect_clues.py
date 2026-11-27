"""Count observable opportunities at logged upper-level decisions, after acceptance."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math


def load(path):
    return json.loads(path.read_bytes())


def events(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def summarize(record):
    path = Path(record["result"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["result_sha256"]
    decisions = events(path.parent / "agent/decision_trace.jsonl")
    checks = events(path.parent / "agent/confirm_events.jsonl")
    counts = Counter()
    registered = set()
    rejected_ids = set()
    for decision in decisions:
        snapshot = decision.get("agent_snapshot")
        if snapshot is None:
            continue
        step = int(decision["step"])
        state = decision["world_state"]
        instances = snapshot["instances"]
        registered.update(n["instance_id"] for n in instances)
        # Only grades already logged on earlier steps are attributed here.
        grades = {r["observation"]:r["trusted"] for r in checks
                  if r["event"]=="reference_quality" and r["step"] < step}
        promoted = {r["instance"] for r in checks if r["event"]=="weak_reference_registered" and r["step"]<step}
        entries = {str(n["id"]):n for n in state.get("instances",[])}
        rejected = [n for n in instances if n["observation_ids"] and not n["reported"]
                    and n["instance_id"] not in promoted
                    and all(grades.get(oid) is False for oid in n["observation_ids"])]
        near = []
        for n in rejected:
            entry = entries.get(str(n["instance_id"]), {})
            distance = entry.get("dist_m")
            if isinstance(distance,(int,float)) and math.isfinite(distance) and .1 <= distance <= 1.8:
                near.append(n)
        rejected_ids.update(n["instance_id"] for n in rejected)
        counts["decisions_with_snapshot"] += 1
        counts["with_any_registration"] += bool(instances)
        counts["with_unreported_registration"] += any(not n["reported"] for n in instances)
        counts["with_any_local_class_proposal"] += bool((state.get("local_rgb_proposals") or {}).get("boxes"))
        counts["with_unreported_graded_weak_registration"] += bool(rejected)
        counts["with_graded_weak_registration_in_displayed_range"] += bool(near)
        counts["with_reported_registration"] += any(n["reported"] for n in instances)
    return dict(scope="Counts logged decision snapshots only. A weak registration has all its observations explicitly rejected on previous steps. Absence of logged evidence is unknown. Map-provided dist_m is rounded. These are opportunities, not labels or guaranteed planner triggers.",
                decision_counts=dict(counts), total_distinct_internal_registrations=len(registered),
                distinct_internal_registrations_with_explicitly_rejected_references=len(rejected_ids),
                source_result_sha256=record["result_sha256"])


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    folder = here.parent / "mapping_runtime/pilots/identity_stage8_weak_relook"
    assert (folder / "closed.json").is_file()
    ticket, = (folder/"comparison/accepted/second").glob("*.json")
    receipt = load(ticket)
    result = summarize(receipt)
    panel_row, = load(folder / "panel.json")["rows"]
    baseline_receipt = load(Path(panel_row["ref"]))
    baseline_path = Path(baseline_receipt["result"])
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == panel_row["result_sha256"]
    base, = load(baseline_path)["episodes"]
    run_path = Path(receipt["result"])
    current, = load(run_path)["episodes"]
    assert current["episode_id"] == base["episode_id"]
    baseline_targets = set(map(int, base["reached_targets"]))
    result["same_credited_targets_as_original"] = baseline_targets == set(map(int, current["reached_targets"]))
    result["original_credited_clock_already_reported_at_capture"] = [
        dict(step=r["step"], previously_credited=bool(baseline_targets.intersection(
            int(t) for event in current["report_events"] if event["step"]<r["step"] for t in event["new_targets"]
        ))) for r in events(run_path.parent / "agent/confirm_events.jsonl") if r["event"]=="weak_relook_capture"
    ]
    dest=here/"clue_summary.json"
    if dest.exists():
        old = load(dest)
        assert all(result.get(k)==v for k,v in old.items())
    dest.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["decision_counts"]))
