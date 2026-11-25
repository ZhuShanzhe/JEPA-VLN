"""Read accepted frontier decisions, verified maps and execution receipts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_bytes())


def analyze(project, physical=None):
    pilot=project/"mapping_runtime/pilots/jepa_confirm_v18_first2"
    cases=[]
    for row in read(pilot/"panel.json")["rows"]:
        tag=row["ordinal"]
        receipt=pilot/"comparison/accepted/second"/(row["episode_id"]+".json")
        if not receipt.exists():
            cases.append(dict(ordinal=tag,status="awaiting_acceptance"))
            continue
        ref=read(receipt)
        path=Path(ref["result"])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==ref["result_sha256"]
        ep,=read(path)["episodes"]
        assert ep["max_steps_budget"]==300
        ev_path=path.parent/"agent/v13_routes.jsonl"
        events=[json.loads(s) for s in ev_path.read_bytes().splitlines()] if ev_path.exists() else []
        by=lambda t:[e for e in events if e["event"]==t]
        filter_table=[]
        geo=Counter()
        for e in by("view_filter"):
            ids=e["ids"]
            rows=e["view_profiles"]
            assert not rows or len(ids)==len(rows)
            for r in rows:
                geo["offered_prefixes_with_map"]+=1
                geo["nominal_prefix_truncated_by_map"]+=r["reach_check"]=="movement_leaves_traversable_map"
                geo["fully_admissible_for_ranking"]+=r["allowed"]
                geo["eligible_prefix_exposes_new_angles"]+=r["allowed"] and r["new_view_fraction"]>0
            filter_table.append(dict(step=e["step"],offered=len(ids),checked=len(rows),
                                     any_viable_alternative=e["viable_alternative"],
                                     passing_alternatives=sum(r["allowed"] for r in rows[1:]),
                                     truncation_sources=dict(Counter(r["reach_check"] for r in rows))))
        choices=[]
        for e in by("rank"):
            i=e["index"]
            assert e["ids"][i]==e["selected"]
            profiles=e.get("view_profiles",[])
            choices.append(dict(step=e["step"],candidates=len(e["ids"]),selected_native=e["original"]==e["selected"],
                                map_changed_choice=e.get("pre_view_index",i)!=i,
                                predicted_new_angle_share=profiles[i]["new_view_fraction"] if profiles else None,
                                available_view_steps=profiles[i]["view_steps"] if profiles else None,
                                effective_evidence_rate=round(float(e["evidence_rate"][i]),4),
                                map_reason=e.get("view_reason")))
        summary=dict(ordinal=tag,status="accepted",steps=ep["steps_taken"],checks=len(by("eligibility")),
                          gate_reasons=dict(Counter(str(e["reason"] or "geometrically_distinct") for e in by("eligibility"))),
                          map_gate=filter_table,geometry_counts=dict(geo),selections=choices,
                          view_track_starts=len(by("view_track_start")),
                          view_track_ends=dict(Counter(e["reason"] for e in by("view_track_end"))),
                          first_different_prefix_ends=dict(Counter(e["reason"] for e in by("commit_end"))),
                          changed_from_native=sum(not s["selected_native"] for s in choices),
                          changed_by_map_scoring=sum(s["map_changed_choice"] for s in choices))
        if physical is not None:
            case, = [r for r in physical if r["ordinal"]==tag]
            summary["physical_view_checks"]=case["view_execution"]
            assert case["view_execution"]["selected_views"]==summary["view_track_starts"]
        cases.append(summary)
    return cases


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project",type=Path)
    parser.add_argument("--physical",type=Path,help="Accepted opportunities.py aggregate after independent Habitat replay")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=analyze(args.project,read(args.physical) if args.physical else None)
    if args.output:
        assert all(c["status"]=="accepted" for c in result)
        args.output.write_text(json.dumps(result,indent=2)+"\n")
    else:
        print(json.dumps(result,indent=2))
