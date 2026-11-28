"""Independently render and execute all selected original/new paths in the fresh blind houses."""
from collections import Counter
import json
import numpy as np
from sample_views import HERE, simulator, camera, SOURCE_H, ACTION, ASSETS, inventory


def main():
    records=json.loads((HERE/"private_evaluation/blind_choices.json").read_text())
    counts=Counter()
    for house in sorted({r["house"] for r in records}):
        sim=simulator(house)
        try:
            inventory(sim,ASSETS)
            ag=sim.get_agent(0)
            folder=HERE/"private_render_blind60"/("blind_"+house)
            with np.load(folder/"controls.npz") as raw:
                d={k:raw[k] for k in ("action","group","pixels","first_blocked","reach","pose","budget","end_error_m","start_pixels","scan_pixels")}
            for r in records:
                if r["house"]!=house:continue
                with np.load(folder/f"{r['source']}_sensor.npz") as s:
                    start,yaw,oid=s["position"],float(s["yaw"]),int(s["oid"])
                    expected=s["depth"]
                    expected_left=s["left_depth"]
                ix=int(np.flatnonzero(d["group"]==r["group"])[0])
                start_obs=camera(sim,start,yaw,SOURCE_H,0.)
                assert abs(int((start_obs["ids"]==oid).sum())-int(d["start_pixels"][ix]))<=1
                ag.act(ACTION[2])
                left=sim.get_sensor_observations()
                assert np.array_equal(left["depth"].squeeze(),expected_left)
                assert abs(int((left["ids"]==oid).sum())-int(d["scan_pixels"][ix]))<=1
                ag.act(ACTION[3])
                assert np.array_equal(sim.get_sensor_observations()["depth"].squeeze(),expected)
                state=ag.get_state()
                counts["groups"]+=1
                for arm in ("old","jepa"):
                    i=r[arm]
                    counts[arm+"_actions"]+=2
                    if i is None:
                        counts[arm+"_deferred"]+=1
                        continue
                    assert d["group"][i]==r["group"]
                    ag.set_state(state,reset_sensors=False,infer_sensor_states=False)
                    plan=d["action"][i];plan=plan[plan>0]
                    assert len(plan)+2<=d["budget"][i]
                    blocked=0;px=-1
                    for k,a in enumerate(plan,1):
                        origin=ag.state.position.copy()
                        hit=ag.act(ACTION[a]);counts[arm+"_actions"]+=1
                        if a==1 and (hit or np.linalg.norm((ag.state.position-origin)[[0,2]])<.23):
                            blocked=k;break
                    dx,dz=d["pose"][i,:2]
                    endpoint=start+np.asarray([dx*np.cos(yaw)-dz*np.sin(yaw),0.,-dx*np.sin(yaw)-dz*np.cos(yaw)])
                    err=np.linalg.norm((ag.state.position-endpoint)[[0,2]])
                    assert abs(err-d["end_error_m"][i])<.001
                    reached=blocked==0 and err<.14
                    if reached:px=int((sim.get_sensor_observations()["ids"]==oid).sum())
                    assert reached==bool(d["reach"][i]) and blocked==d["first_blocked"][i]
                    assert abs(px-int(d["pixels"][i]))<=1
                    counts[arm+"_reached"]+=int(reached)
                    counts[arm+"_clear"]+=int(reached and px>=369)
                    counts[arm+"_issued"]+=1
        finally:sim.close()
    result=json.loads((HERE/"blind_evaluation.json").read_text())["subsets"]["all"]
    for arm in ("old","jepa"):
        for key in ("actions","clear"):
            assert counts[arm+"_"+key]==result[arm].get(key,0)
    receipt=dict(checks="Source and scan instance pixels, two depth frames, contact index, final position error and final instance pixels matched independent physical rollouts.",counts=counts)
    (HERE/"selected_replay.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(receipt,flush=True)


if __name__=="__main__":main()
