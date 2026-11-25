"""Join frozen native references, accepted treatments, independent physical replay and module events."""
from collections import Counter
import json
from pathlib import Path
import sys


def load(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(x) for x in path.read_bytes().splitlines() if x]


def arm(ref, replay_path, quota, diag):
    metrics = diag.accepted(ref, replay_path, quota, False)
    result = Path(load(ref)["result"])
    doc = load(result)
    episode, = doc["episodes"]
    replay = load(replay_path)
    reports = episode["report_events"]
    found = set()
    for event in reports:
        newly = set(event["new_targets"])
        assert not (found & newly)
        found.update(newly)
    assert len(found) == metrics["found"] and bool(len(found)>=quota)==metrics["success"]
    trace = episode["paper"]["agent_trace"]
    expose = [v["exposure"]["moderate"]["first_qualifying_step"] for v in replay["targets"]]
    metrics["first_clear_view_steps"] = sorted(int(x) for x in expose if x is not None)
    metrics["report_timing"] = [{"step":e["step"],"new_count":len(e["new_targets"]),
                                 "near_count":len(e["near_targets"])} for e in reports]
    metrics["validated_decisions"] = dict(Counter((d.get("output") or {}).get("action","unknown")
                                         for d in trace["decisions"] if d.get("validation")=="ok"))
    return metrics, doc["manifest"]["arguments"], episode, result.parent/"agent"


def route_diagnostics(folder, episode):
    events = lines(folder/"v13_routes.jsonl")
    ranks = [e for e in events if e["event"]=="rank"]
    physical = [e for e in events if e["event"]=="commit_end" and e["reason"]=="first_difference_observed"]
    for r in ranks:
        assert r["ids"][r["index"]]==r["selected"] and len(r["ids"])>=2
    return dict(eligibility=dict(Counter(str(e["reason"] or "eligible") for e in events if e["event"]=="eligibility")),
                event_counts=dict(Counter(e["event"] for e in events)),
                rank_steps=[r["step"] for r in ranks], ranked_group_sizes=[len(r["ids"]) for r in ranks],
                non_native_selections=sum(r["selected"]!=r["original"] for r in ranks),
                ack_first_different_primitive=len(physical),
                commit_end_reasons=dict(Counter(e["reason"] for e in events if e["event"]=="commit_end")),
                observation_scan_actions_acknowledged=sum(e.get("physical_turns",0) for e in events if e["event"]=="scan_end"))


def main(root, output):
    sys.path.insert(0,str(root/"jepa_env_v13"))
    import diagnose_full_most as diag
    spec=load(Path(__file__).with_name("design.json"))
    pilot=root/"mapping_runtime/pilots/jepa_confirm_v18_first2"
    native_replay=root/"mapping_runtime/analysis/visibility/native_20261115/verified"
    panel=load(pilot/"panel.json")["rows"]
    assert [(r["ordinal"],r["quota"]) for r in panel]==[(t["ordinal"],t["quota"]) for t in spec["tasks"]]
    rows=[]
    for r in panel:
        file=r["episode_id"]+".json"
        ref=Path(r["ref"])
        assert load(ref)["result_sha256"]==r["result_sha256"]
        b,ba,be,_=arm(ref,native_replay/file,r["quota"],diag)
        n,na,ne,folder=arm(pilot/"comparison/accepted/second"/file,pilot/"replay"/file,r["quota"],diag)
        for k in ("threshold","goal_type","action_protocol","instruction_policy","pool_match_threshold",
                  "same_floor_only","same_floor_height_threshold"):
            assert ba[k]==na[k], k
        for k in ("episode_id","goal_text","target_mode","max_steps_budget","required_target_count","sensor_hfov"):
            assert be[k]==ne[k], k
        bd,nd=be["paper"]["episode_definition"],ne["paper"]["episode_definition"]
        for k in ("start_position","start_rotation","target_object_ids","target_positions","view_points","query_program"):
            assert bd[k]==nd[k], k
        row={"ordinal":r["ordinal"],"quota":r["quota"],"native":b,"treatment":n,
             "route":route_diagnostics(folder,ne), "excluded_engineering_attempts":diag.quarantines(pilot,r["episode_id"])}
        rows.append(row)
    closed=load(pilot/"five_comparison.json")
    billed=closed["tagged_decision_accounting"]
    out={"cases":rows,"totals":{},"accounting":{key:billed[key] for key in
         ("events","reported_tokens","known_currency_amount","shared_caption_ledger_excluded_from_decision_usage")}}
    for method in ("native","treatment"):
        out["totals"][method]={key:sum(r[method][key] for r in rows) for key in ("found","capped_found","success","steps")}
    output.write_text(json.dumps(out,allow_nan=False,indent=2)+"\n")
    print(json.dumps(out["totals"]))


if __name__=="__main__":
    main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]))
