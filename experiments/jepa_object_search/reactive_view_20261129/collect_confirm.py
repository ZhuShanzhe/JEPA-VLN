"""Physical RGB-D time samples for the verifier. Semantics are offline labels only."""
import argparse
from pathlib import Path
import json,sys,os
import numpy as np
HERE=Path(__file__).resolve().parent
PRE=HERE.parent/"new_view_20261129"
sys.path.insert(0,str(PRE))
from sample_views import simulator,inventory,ASSETS,camera,SOURCE_H,ACTION

def collect(split,house):
    raw=PRE/"private_render"/(split+"_"+house)
    out=HERE/"private_confirm_data"/split/house
    out.mkdir(parents=True,exist_ok=True)
    if (out/"complete").exists():return
    sim=simulator(house)
    try:
        inventory(sim,ASSETS);agent=sim.get_agent(0)
        for sensor in sorted(raw.glob("*_sensor.npz")):
            num=int(sensor.name.split("_")[0])
            with np.load(sensor) as d:
                pos=d["position"];yaw=float(d["yaw"]);oid=int(d["oid"]);box=d["box"]
            obs=camera(sim,pos,yaw,SOURCE_H,0.);st=agent.get_state()
            frames=[];seen=set()
            def add(theta):
                xyz=agent.state.position-pos
                rot=(float(xyz[0]*np.cos(yaw)-xyz[2]*np.sin(yaw)),float(-xyz[0]*np.sin(yaw)-xyz[2]*np.cos(yaw)),theta)
                k=tuple(round(v,2) for v in rot)
                if k in seen:return
                seen.add(k)
                s=sim.get_sensor_observations()
                frames.append((np.copy(s["rgb"][...,:3]),np.copy(s["depth"].squeeze()).astype(np.float16),np.packbits(s["ids"]==oid),rot))
            add(0.)
            for plan in ( [1]*8, [2,2,1,1,1,1,1,3,3,3,1,1], [3,3,1,1,1,1,1,2,2,2,1,1], [3]*5):
                agent.set_state(st,reset_sensors=False,infer_sensor_states=False);theta=0.
                for k,a in enumerate(plan):
                    collided=agent.act(ACTION[a]);theta+=((a==2)-(a==3))*np.pi/6
                    if k in (0,2,4,6,8,len(plan)-1) or collided:add(theta)
                    if collided:break
            rgb,dep,mask,pose=map(np.asarray,zip(*frames))
            dst=out/(str(num)+".npz")
            tmp=dst.with_suffix(".partial.npz")
            np.savez_compressed(tmp,rgb=rgb,depth=dep,mask=mask,pose=pose,box=box);tmp.replace(dst)
        (out/"complete").touch()
    finally:sim.close()

if __name__=="__main__":
    import argparse
    os.umask(0o077)
    p=argparse.ArgumentParser();p.add_argument("worker",type=int);p.add_argument("workers",type=int);a=p.parse_args()
    folds=json.loads((PRE/"view_design.json").read_text())["cohorts"]
    jobs=[(s,h) for s in ("train","val") for h in folds[s]]
    for split,house in jobs[a.worker::a.workers]:collect(split,house)
