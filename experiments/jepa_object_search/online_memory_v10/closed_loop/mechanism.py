"""Summarize actual route execution and official reports after independent Habitat replay."""
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_bytes())


def events(path):
    return [json.loads(x) for x in path.read_bytes().splitlines()] if path.exists() else []


def selected_plan(row,name):
    if name==row["original"]:
        return row["original_nominal_actions"]
    return [action+1 for action in row["supported_plans"][name]]


def summarize(d):
    comparisons=read(d/"eight_comparison.json")
    audits=read(d/"motion_audit.json")["cases"]
    mode=read(d/"arm.json")["memory_arm"]
    result=[]
    for row,case,post in zip(read(d/"panel.json")["rows"],comparisons["cases"],audits):
        assert row["ordinal"]==case["ordinal"]==post["ordinal"]
        record=read(d/"comparison/accepted/second"/(row["episode_id"]+".json"))
        path=Path(record["result"])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record["result_sha256"]
        ep,=read(path)["episodes"]
        actions=[r["action"] for r in ep["paper"]["steps"]]
        folder=path.parent/"agent"
        obs=events(folder/"memory_updates.jsonl")
        assert [r["step"] for r in obs]==list(range(len(actions)))
        assert [r["step"] for r in obs if r["reset"]]==[0]
        assert all(r["visual_score"] is None or 0<=r["visual_score"]<=1 for r in obs)
        assert (folder/"v10_memory.json").is_file()
        selections=events(folder/"jepa_selection.jsonl")
        active=[r for r in events(folder/"frontier_probe.jsonl") if r["event"]=="execution_selection"]
        assert len(selections)==len(active)
        gap={k:dict(different=0,applied=0,first_different_command_executed=0) for k in ["v8","frozen","no_jepa"]}
        for s,a in zip(selections,active):
            if "scores" not in s:continue
            t=s["step"]
            assert t==a["step"]==s["memory_step"] and max(s["source_steps"])<=t
            assert s["selected"]==s[("full" if mode=="full" else "frozen")+"_selected"]
            assert a["executed"]==s["selected"]
            for k,v in gap.items():
                ref=s[k+"_selected"]
                if s["selected"]==ref:continue
                v["different"]+=1
                if not a["applied"]:continue
                v["applied"]+=1
                take=s["selected_nominal_actions"]; alternative=selected_plan(s,ref)
                delta=next(i for i,(x,y) in enumerate(zip(take,alternative)) if x!=y)
                # Exclude preemption prior to the first divergent command.
                trace=ep["paper"]["agent_trace"]["decisions"]
                interrupted=any(t<e["step"]<=t+delta for e in trace)
                if not interrupted and actions[t:t+delta+1]==take[:delta+1]:
                    v["first_different_command_executed"]+=1
        hits=ep["report_events"]
        result.append(dict(ordinal=case["ordinal"],quota=case["quota"],found=case["matched"],success=case["success"],
            steps=len(actions),reports=len(hits),repeat_reports=sum(bool(h["near_targets"]) and not h["new_targets"] for h in hits),
            empty_reports=sum(not h["near_targets"] for h in hits),eligible_anywhere=post["all_route_eligible_instances"],
            distinct_moderate_exposures=read(d/"replay"/(row["episode_id"]+".json"))["count_exposed"]["moderate"],
            observed_updates=len(obs),visual_measurements=sum(r["measured_rgb"] for r in obs),
            failed_forward_feedback=sum(r["collided"] for r in obs),scored=case["scored"],counterfactuals=gap,
            native_reference_instances=case["found_native"],http_verified=case["verified_gateway_responses"],
            run_quarantines=case["quarantined"],independent_physical_replay=True))
    dest=d/"mechanism.json"
    output=dict(arm=mode,model=comparisons["model"],cases=result)
    if dest.exists():assert read(dest)==output
    else:dest.write_text(json.dumps(output,indent=2))
    print("verified",mode,len(result),flush=True)

if __name__=="__main__":summarize(Path(sys.argv[1]))
