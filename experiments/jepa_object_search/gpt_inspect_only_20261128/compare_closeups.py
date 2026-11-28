"""Measure where enabling the visual trigger changed the paired physical outcome."""
from collections import Counter
import json
from pathlib import Path


def main():
    here=Path(__file__).resolve().parent
    controls=json.loads((here/"parity.json").read_text())["houses"]
    summary=[]
    for i,off in enumerate(controls):
        on=json.loads((here/"runs"/f"{i}_inspect.json").read_text())
        assert off["scene"]==on["scene"] and off["steps"]==on["steps"]==300
        events=[json.loads(l) for l in (here/"private_events"/f"{i}.jsonl").read_text().splitlines()]
        local=[r for r in events if r.get("action_source")=="local" and r.get("last_step",0)>r["step"]]
        requested=[r for r in events if r.get("gpt_mode")=="inspect"]
        first=min((r["step"] for r in local),default=None)
        first_hit_while_inspecting=sum(any(r["step"]<s<=r["last_step"] for r in local) for s in on["first_steps"])
        outcome_diff=on["distinct_approached"]-off["distinct_approached"]
        matched_prefix = on["first_steps"]==off["first_steps"] if first is None else (
            [s for s in on["first_steps"] if s<=first] == [s for s in off["first_steps"] if s<=first])
        summary.append(dict(goal=off["goal"],requested=len(requested),request_outcomes=dict(Counter(e["exec_status"] for e in requested)),
                            first_local_segment_begins_after_step=first,local_physical_actions=sum(r["last_step"]-r["step"] for r in local),
                            local_first_instance_hits=first_hit_while_inspecting,
                            first_hit_steps_agree_before_first_local_action=matched_prefix,
                            diff_in_full_episode_instances=outcome_diff))
    report=dict(paired_houses=summary,any_local_movement=sum(r["local_physical_actions"]>0 for r in summary),
                local_first_instance_hits=sum(r["local_first_instance_hits"] for r in summary))
    (here/"closeup_comparison.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
