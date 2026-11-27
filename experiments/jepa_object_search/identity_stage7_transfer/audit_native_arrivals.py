"""Join accepted result receipts to contemporaneous decision events; release aggregates only."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def audit(campaign):
    decisions = Counter()
    outcomes = defaultdict(Counter)
    report_counts = Counter()
    provenance = []
    receipts = sorted((campaign / "accepted/native").glob("*.json"))
    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_bytes())
        path = Path(receipt["result"])
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == receipt["result_sha256"]
        episode, = json.loads(raw)["episodes"]
        assert episode["target_mode"] == "many"
        log = (path.parent / "agent/decision_trace.jsonl").read_bytes()
        provenance.append(digest + ":" + hashlib.sha256(log).hexdigest())
        trace = [json.loads(row) for row in log.splitlines() if row.strip()]
        requests = defaultdict(list)
        for dec in trace:
            action = (dec.get("output") or {}).get("action")
            if dec["event"] == "arrival":
                decisions[str(action)] += 1
            if action == "REPORT_FOUND":
                requests[dec["step"]].append(dec)
        for report in episode["report_events"]:
            step = report["step"]
            assert len(requests[step]) == 1
            dec, = requests[step]
            where = "arrival" if dec["event"] == "arrival" else "other"
            counts = outcomes[where]
            counts["issued_reports"] += 1
            new = set(report["new_targets"])
            previous = set(report["near_targets"]) - new
            if new:
                counts["at_least_one_new"] += 1
            elif previous:
                counts["only_previously_claimed"] += 1
            else:
                counts["no_target_in_range"] += 1
            if new and previous:
                counts["both_new_and_previously_claimed"] += 1
            if report["is_false_positive"]:
                counts["official_false_positive_flag"] += 1
            if new and report["is_false_positive"]:
                counts["new_and_official_false_positive"] += 1
            counts["zero_tools_in_report_turn"] += int(dec["output"]["tool_calls"] == 0)
        report_counts[len(episode["report_events"])] += 1
    for counts in outcomes.values():
        assert counts["issued_reports"] == sum(counts[k] for k in (
            "at_least_one_new","only_previously_claimed","no_target_in_range"))
        for key in ("both_new_and_previously_claimed","new_and_official_false_positive"):
            counts.setdefault(key,0)
    return {
        "generated_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source":"native_20261115, accepted/native result receipts, per-action report_events and per-step decision_trace",
        "sample":"Every accepted native-arm episode available at audit time; partial non-random snapshot of 300 scheduled development episodes.",
        "accepted_episodes":len(receipts),
        "episodes_without_report":report_counts[0],
        "source_result_and_decision_set_sha256":hashlib.sha256("\n".join(sorted(provenance)).encode()).hexdigest(),
        "arrival_decision_counts":dict(decisions), "by_report_origin":dict(outcomes),
        "unit":"Each action counted once. At least one new includes mixed actions. Only previously claimed can include tolerated repeat or over-quota claims. False-positive flags can overlap a new claim under the official multi-instance scorer. These are not success rates.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.campaign),indent=2))
