"""Summarize private event logs without publishing coordinates, actions, or trajectories."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import sys


def diagnose(stage):
    panel=json.loads((stage/"panel.json").read_text())
    summaries=[]
    totals=Counter()
    holders=Counter()
    releases=Counter()
    surfaces=Counter()
    gates=defaultdict(Counter)
    for i,spec in enumerate(panel):
        result=json.loads((stage/"runs"/f"{i}_local.json").read_text())
        previous=json.loads((stage/"runs"/f"{i}_new.json").read_text())
        events=[json.loads(l) for l in (stage/"private_events"/f"{i}.jsonl").read_text().splitlines()]
        seen={}
        ages=Counter()
        arrivals=set()
        reduced_distance=Counter()
        first_hits=Counter()
        for step in result["first_steps"]:
            if step==0:
                first_hits["start"]+=1
                continue
            segments=[e["action_source"] for e in events if "last_step" in e
                      and e["step"] < step <= e["last_step"]]
            if len(segments)!=1:
                raise ValueError("first physical hit not assigned to one executed segment")
            first_hits[segments[0]]+=1
        for ev in events:
            gid=ev.get("goal_id")
            if ev.get("event")=="arrived":
                assert gid in seen and gid not in arrivals
                arrivals.add(gid)
                continue
            rel=ev.get("released")
            if rel and rel["event"]=="arrived":
                assert rel["goal_id"] in seen and rel["goal_id"] not in arrivals
                arrivals.add(rel["goal_id"])
            if ev.get("action_source")!="local":
                continue
            sig=(ev["site_xy"],ev["site_yaw"],ev["anchor_xy"])
            if ev["event"]=="acquired":
                assert gid not in seen
                seen[gid]=(sig,ev["distance_m"])
            else:
                assert ev["event"]=="retained" and seen[gid][0]==sig
                if ev["distance_m"]<seen[gid][1]-.1:
                    reduced_distance[gid]+=1
            ages[gid]+=1
            assert ages[gid]==ev["age"]
        stats=result["stats"]
        assert len(arrivals)==result["counts"]["releases"].get("arrived",0)
        assert sum(stats.get(k+"_steps",0) for k in ("scan","region","local","tail","empty"))==300
        assert len(seen)==stats.get("acquired",0)
        for k,v in stats.items():
            if k!="max_age": totals[k]+=v
        holders.update(ages.values())
        releases.update(result["counts"]["releases"])
        surfaces.update(result["counts"]["surfaces"])
        for g,cutoffs in result["distance_by_phase"].items():gates[g].update(cutoffs)
        summaries.append(dict(goal=spec["goal"],old_instances=previous["distinct_approached"],local_instances=result["distinct_approached"],
                              initial_instances=result["already_at_start"],
                              old_half=previous["approached_half_m"],local_half=result["approached_half_m"],
                              old_one=previous["approached_one_m"],local_one=result["approached_one_m"],
                              first_approach_by_phase=dict(first_hits),acquired=len(seen),retained_multiple=sum(a>=2 for a in ages.values()),
                              closed_distance_on_recheck=len(reduced_distance),arrived=len(arrivals),gates=result["distance_by_phase"]))
    return dict(houses=summaries,local_decision_totals=dict(totals),decisions_per_local_goal=dict(holders),
                local_release_reasons=dict(releases),selected_surface_labels=dict(surfaces),
                sum_house_distinct_instances_by_phase={k:dict(v) for k,v in gates.items()})


if __name__=="__main__":
    stage=Path(sys.argv[1])
    (stage/"decision_summary.json").write_text(json.dumps(diagnose(stage),indent=2)+"\n")
