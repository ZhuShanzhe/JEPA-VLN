"""Counterfactual camera pitch at the same position: never used as an action in the scored policy."""
import argparse
from collections import Counter
import json
import math
import numpy as np
from approach_probe import (HERE, PREVIOUS, SOURCES, simulator, inventory, ASSETS, camera,
                            ACTION, SOURCE_H, angle, image_info, rotate, world_points, lost_projection)


def measure_pitch(sim,ag,oid):
    state=ag.get_state()
    pos=state.position.copy();yaw=angle(ag)
    out={}
    for q in (-30,-15,0,15,30):
        obs=camera(sim,pos,yaw,SOURCE_H,q)
        out[q]=int((obs["ids"]==oid).sum())
    ag.set_state(state,reset_sensors=False,infer_sensor_states=False)
    return out


def run(i):
    h=json.loads((SOURCES/"view_design.json").read_text())["cohorts"]["blind"][i]
    sim=simulator(h)
    counts=Counter()
    try:
        inventory(sim,ASSETS);ag=sim.get_agent(0)
        for f in (PREVIOUS/"private_roll"/h).glob("*.json"):
            row=json.loads(f.read_text());ev=row["events"]
            if not row["focus"] or row["new_confirmation"] is not None or max(e["px"] for e in ev)>=369:continue
            k=int(np.argmax([e["px"] for e in ev]))
            with np.load(SOURCES/"private_render_blind60"/("blind_"+h)/(str(row["source"])+"_sensor.npz")) as d:
                st,a,oid=d["position"],float(d["yaw"]),int(d["oid"])
            camera(sim,st,a,SOURCE_H,0)
            for e in ev[1:k+1]:ag.act(ACTION[e["action"]])
            at_peak=measure_pitch(sim,ag,oid)
            assert at_peak[0]==ev[k]["px"]
            counts["groups"]+=1
            counts["pitch_alone_reaches_369_at_original_peak"]+=int(max(at_peak.values())>=369)
            counts["pitch_changes_visible_area_by_at_least_50pct_at_peak"]+=int(max(at_peak.values())>=1.5*at_peak[0])
            _,xx,dep=image_info(sim,oid)
            right=(xx+.5-96)*dep/96
            aim=int(round(-math.atan2(np.median(right),np.median(dep))/(math.pi/6)))
            rotate(ag,aim);initial=image_info(sim,oid)[0]
            points=world_points(sim,ag,oid)
            p=ag.state.position.copy()
            hit=ag.act(ACTION[1])
            if hit or np.linalg.norm((ag.state.position-p)[[0,2]])<.23:continue
            now=image_info(sim,oid)[0]
            detail=lost_projection(sim,ag,oid,points)
            if now<initial and detail and detail["outside_vertical"]>.5*detail["source_points"]:
                counts["direct_first_step_loses_majority_vertically"]+=1
                tilted=measure_pitch(sim,ag,oid);best=max(tilted.values())
                counts["lost_majority_vertically_tilt_recovers_initial_area"]+=int(best>=initial)
                counts["lost_majority_vertically_tilt_reaches_369"]+=int(best>=369)
                counts["lost_majority_vertically_tilt_increases_area"]+=int(best>now)
    finally:sim.close()
    out=HERE/"private_approach"/("pitch_"+str(i)+".json")
    out.write_text(json.dumps(counts))


def main():
    result=sum([Counter(json.loads((HERE/"private_approach"/("pitch_"+str(i)+".json")).read_text())) for i in range(3)],Counter())
    for name in ("pitch_alone_reaches_369_at_original_peak","lost_majority_vertically_tilt_reaches_369"):
        result.setdefault(name,0)
    assert result["groups"]==33 and result["direct_first_step_loses_majority_vertically"]==4
    msg=dict(counts=dict(result),protocol="At the earliest peak frame in each of the 33 previously unreported never-clear groups, rerender simulated camera at -30,-15,0,+15,+30 degrees pitch without changing position, azimuth, FOV or pixel dimensions. Repeat at +0.25m in four matched cases where most tracked target points exit vertically. Oracle maximum over five pitches, outside controller action space; not a policy score.")
    (HERE/"approach_pitch.json").write_text(json.dumps(msg,indent=2)+"\n")
    print(msg)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int)
    i=p.parse_args().index
    if i is None:main()
    else:run(i)
