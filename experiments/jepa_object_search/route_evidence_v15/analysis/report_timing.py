"""Summarize search opportunities, report timing and the first intervention."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_bytes())


def one(ref_path, replay_path):
    receipt = read(ref_path)
    result = Path(receipt["result"])
    assert hashlib.sha256(result.read_bytes()).hexdigest() == receipt["result_sha256"]
    replay = read(replay_path)
    assert replay["source_result_sha256"] == receipt["result_sha256"]
    ep, = read(result)["episodes"]
    trace = ep["paper"]["steps"]
    assert len(trace) == replay["steps_replayed"]
    targets = replay["targets"]
    exposure = sorted(t["exposure"]["moderate"]["first_qualifying_step"] for t in targets
                      if t["exposure"]["moderate"]["first_qualifying_step"] is not None)
    decisions = [d for d in ep["paper"]["agent_trace"]["decisions"] if d.get("validation")=="ok"]
    terminal = (decisions[-1].get("output") or {}) if decisions else {}
    events = ep["report_events"]
    return dict(steps=len(trace), termination_action=trace[-1]["action_role"], agent_finished=bool(ep.get("finished_by_agent")),
                last_validated_decision=terminal.get("action"),
                clear_exposures=exposure,
                approached=len(ep["approached_target_ids"]),
                reports=[dict(step=e["step"], new_instances=len(e["new_targets"])) for e in events],
                capped_hits=min(ep["required_target_count"], ep["found_count"]),
                successful_decision_counts=dict(Counter((d.get("output") or {}).get("action", "unknown") for d in decisions)))


def main(project, destination):
    pilots = project/"mapping_runtime/pilots"
    pilot=pilots/"jepa_evidence_v15_first2"
    v13=pilots/"jepa_round13_horizon_five"
    values=[]
    for row in read(pilot/"panel.json")["rows"]:
        name=row["episode_id"]+".json"
        candidates = [(Path(row["ref"]), project/"mapping_runtime/analysis/visibility/native_20261115/verified"/name),
                      (v13/"comparison/accepted/second"/name,v13/"replay"/name),
                      (pilot/"comparison/accepted/second"/name,pilot/"replay"/name)]
        if not candidates[-1][1].is_file():
            continue
        arms = {arm:one(ref, replay) for arm,(ref,replay) in zip(("native","v13","v15"),candidates)}
        original, = read(Path(read(candidates[1][0])["result"]))["episodes"]
        test_result=Path(read(candidates[2][0])["result"])
        tested, = read(test_result)["episodes"]
        a,b=original["paper"]["steps"],tested["paper"]["steps"]
        first = next((i for i,(x,y) in enumerate(zip(a,b)) if x["action"]!=y["action"]), min(len(a),len(b)) if len(a)!=len(b) else None)
        route = [json.loads(line) for line in (test_result.parent/"agent/v13_routes.jsonl").read_bytes().splitlines()]
        first_scan=next((r["step"] for r in route if r["event"]=="scan_start"),None)
        values.append(dict(ordinal=row["ordinal"], arms=arms, first_different_action_vs_logged_v13=first,
                           first_v15_scan=first_scan,
                           divergence_before_v15_scan=first is not None and (first_scan is None or first < first_scan)))
    if destination is None:
        print(json.dumps(values))
    else:
        assert len(values)==2
        with destination.open("x") as stream:
            json.dump(values,stream,indent=2)
            stream.write("\n")


if __name__=="__main__":
    main(Path(sys.argv[1]),Path(sys.argv[2]) if len(sys.argv)>2 else None)
