"""Offline counterfactual audit from the old actor's actual stop, within its remaining budget.

This program never selects online actions. Only the offline evaluator reads semantic images.
"""
from collections import Counter
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/"new_view_20261129"
sys.path.insert(0,str(PRIOR))
from sample_views import simulator, camera, inventory, ASSETS, SOURCE_H, ACTION
from view_geometry import turns


def steps(agent,sim,action,instance):
    before=agent.state.position.copy()
    hit=agent.act(ACTION[action])
    stopped=action==1 and (hit or np.linalg.norm((agent.state.position-before)[[0,2]])<.23)
    px=int((sim.get_sensor_observations()["ids"]==instance).sum())
    return bool(stopped),px


def check_house(house):
    private=HERE/"private_stop_probes"
    private.mkdir(mode=0o700,exist_ok=True)
    records=json.loads((PRIOR/"private_evaluation/blind_choices.json").read_text())
    folder=PRIOR/"private_render_blind60"/("blind_"+house)
    with np.load(folder/"controls.npz") as f:
        d={k:f[k] for k in ("group","kind","reach","action","start_pixels","scan_pixels","budget","pose","first_blocked")}
    sim=simulator(house)
    audit=[]
    try:
        inventory(sim,ASSETS)
        ag=sim.get_agent(0)
        for r in (r for r in records if r["house"]==house):
            group=np.flatnonzero(d["group"]==r["group"])
            old=group[d["kind"][group]=="old"]; i=group[0]
            if d["reach"][old].any() or max(d["start_pixels"][i],d["scan_pixels"][i])>=369:continue
            with np.load(folder/f"{r['source']}_sensor.npz") as s:
                start,syaw,oid=s["position"],float(s["yaw"]),int(s["oid"])
                l,t,rr,b=map(int,s["box"]); u,v=(l+rr)/2,(t+b)/2
                patch=s["depth"][int(v)-2:int(v)+3,int(u)-2:int(u)+3]
                z=float(np.median(patch[np.isfinite(patch)&(patch>.05)]))
            x=(u-96)*z/96
            anchor=start+np.array([x*math.cos(syaw)-z*math.sin(syaw),0,-x*math.sin(syaw)-z*math.cos(syaw)])
            camera(sim,start,syaw,SOURCE_H,0.)
            j=r["jepa"]; plan=() if j is None else d["action"][j]; yaw=syaw
            spent=0; peak=0; bump=False
            for act in [2,3,*[int(a) for a in plan if a]]:
                bump,pix=steps(ag,sim,act,oid);spent+=1
                if act in (2,3):yaw+=(1 if act==2 else -1)*math.pi/6
                peak=max(peak,pix)
                if bump:break
            remaining=int(d["budget"][i])-spent
            assert remaining>=0
            row=dict(house=house,source=r["source"],prior_ever_clear=peak>=369,prior_hit=bump,
                     remaining=remaining,probe_opportunity=False,minimum_extra_actions=None,
                     clear_on_inplace_turn=False,clear_on_translation_branch=False)
            stop=ag.get_state()
            if peak>=369 or remaining==0:
                audit.append(row);continue
            # Twelve bearings. At each 0.5m up to 3m, try facing the source-estimated target.
            # Use shared physical prefixes for efficiency. Count every primitive including yaw.
            def note(cost,px,translation):
                if px<369:return
                row["probe_opportunity"]=True
                row["clear_on_translation_branch" if translation else "clear_on_inplace_turn"]=True
                if row["minimum_extra_actions"] is None or cost<row["minimum_extra_actions"]:
                    row["minimum_extra_actions"]=cost
            for bearing in range(-5,7):
                ag.set_state(stop,reset_sensors=False,infer_sensor_states=False)
                rot=turns(bearing)
                if len(rot)>remaining:continue
                cost=0
                for act in rot:
                    _,px=steps(ag,sim,act,oid);cost+=1;note(cost,px,False)
                orientation=yaw+bearing*math.pi/6
                for offset in range(1,min(12,remaining-cost)+1):
                    collision,px=steps(ag,sim,1,oid);cost+=1;note(cost,px,True)
                    if collision:break
                    if offset%2 or cost>=remaining:continue
                    check=ag.get_state()
                    delta=anchor-check.position
                    goal_yaw=math.atan2(-delta[0],-delta[2])
                    adjustment=turns(round((goal_yaw-orientation)/(math.pi/6)))
                    for n,act in enumerate(adjustment[:remaining-cost],1):
                        _,px=steps(ag,sim,act,oid);note(cost+n,px,True)
                    ag.set_state(check,reset_sensors=False,infer_sensor_states=False)
            audit.append(row)
    finally:
        sim.close()
    (private/(house+".json")).write_text(json.dumps(audit))
    print(house,len(audit),flush=True)


def summarize():
    houses=json.loads((PRIOR/"view_design.json").read_text())["cohorts"]["blind"]
    rows=[r for h in houses for r in json.loads((HERE/"private_stop_probes"/(h+".json")).read_text())]
    assert len(rows)==94 and sum(r["prior_ever_clear"] for r in rows)==40
    unresolved=[r for r in rows if not r["prior_ever_clear"]]
    def count(block):
        costs=[r["minimum_extra_actions"] for r in block if r["probe_opportunity"]]
        return dict(groups=len(block),positive_budget=sum(r["remaining"]>0 for r in block),
                    clearable_in_finite_probe=len(costs),min_extra_actions_median=float(np.median(costs)) if costs else None,
                    clearable_by_turning_in_place=sum(r["clear_on_inplace_turn"] for r in block))
    summary=dict(protocol="Offline finite probe from the actual previous stop. At 12 bearings move up to 3m and look towards source-estimated target every 0.5m. Original budget only, first contact stops each tested branch. A positive is evidence of an opportunity; a negative does not rule out different maneuvers.",
                 previous_never_clear=count(unresolved),previous_never_clear_after_contact=count([r for r in unresolved if r["prior_hit"]]))
    (HERE/"after_stop_opportunities.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(summary)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--index",type=int);args=ap.parse_args()
    if args.index is None:summarize()
    else:check_house(json.loads((PRIOR/"view_design.json").read_text())["cohorts"]["blind"][args.index])
