"""Repeat prospectively selected real source scans and branches against payloads without reading policy selections."""
from collections import Counter
import json,math
from pathlib import Path
import numpy as np
from sample_views import HERE,ASSETS,SOURCE_H,camera,simulator,inventory,ACTION

def check(split,homes):
    counts=Counter()
    for house in homes:
        sim=simulator(house); ag=sim.get_agent(0)
        try:
            inventory(sim,ASSETS)
            d=HERE/("private_render_blind60" if split=="blind" else "private_render")/(split+"_"+house)
            with np.load(d/"controls.npz") as data:
                z={k:data[k] for k in ("action","reach","pixels","first_blocked","start_pixels","scan_pixels","source_index","kind","pose","view_id","plan_id","budget")}
            for idx in sorted(set(z["source_index"]))[::11]:
                with np.load(d/f"{idx}_sensor.npz") as source:
                    start=source["position"]; yaw=float(source["yaw"]); oid=int(source["oid"])
                    camera(sim,start,yaw,SOURCE_H,0.)
                    ag.act(ACTION[2]);left=sim.get_sensor_observations()
                    assert np.allclose(left["depth"].squeeze(),source["left_depth"],atol=1e-4)
                    scan_pix=int((left["ids"]==oid).sum())
                    ag.act(ACTION[3]);obs=sim.get_sensor_observations(); state=ag.get_state()
                    assert np.allclose(obs["depth"].squeeze(),source["depth"],atol=1e-4)
                    source_pix=int((obs["ids"]==oid).sum())
                ids=np.flatnonzero(z["source_index"]==idx)
                assert np.all(z["start_pixels"][ids]==source_pix) and np.all(z["scan_pixels"][ids]==scan_pix)
                chosen=list(ids[:7])+list(ids[7::5])
                counts.update(scans=1,scanning_actions=2)
                for i in chosen:
                    ag.set_state(state,reset_sensors=False,infer_sensor_states=False)
                    a=z["action"][i]; col=0; issued=0
                    for step,act in enumerate(a[a>0],1):
                        old=ag.state.position.copy()
                        hit=ag.act(ACTION[int(act)]);issued+=1
                        if act==1 and (hit or np.linalg.norm((ag.state.position-old)[[0,2]])<.23):
                            col=step;break
                    x,m=z["pose"][i,:2]
                    target=start+np.array([x*math.cos(yaw)-m*math.sin(yaw),0.,-x*math.sin(yaw)-m*math.cos(yaw)])
                    r=col==0 and np.linalg.norm((ag.state.position-target)[[0,2]])<.14
                    assert issued+2<=z["budget"][i] and bool(z["reach"][i])==r and col==z["first_blocked"][i]
                    if r:
                        pix=int((sim.get_sensor_observations()["ids"]==oid).sum())
                        assert abs(pix-int(z["pixels"][i]))<=1
                    counts.update(routes=1,route_actions=issued,reached=int(r))
        finally:sim.close()
    return dict(counts)

if __name__=="__main__":
    design=json.loads((HERE/"view_design.json").read_text())["cohorts"]
    metrics={split:check(split,design[split] if split=="blind" else design[split][:1]) for split in design}
    (HERE/"view_physical_replay.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(metrics,flush=True)
