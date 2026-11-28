"""Collect fixed sibling routes by physical replay. All branches share start, target, endpoint and heading."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"pose_horizon_20261128"))
from pose_data import (ASSETS,S,SOURCE_H,camera,choose_sources,features,houses,inventory,simulator,write_rgb)
from path_geometry import from_depth,nominal,path_features,NAMES
RAW=HERE/"private_render"
SAMPLES_PER_HOUSE=22


def options(forwards):
    f=(1,)
    plans=[f*forwards]
    for wait,width in ((0,1),(0,2),(1,2)):
        for left,right in ((2,3),(3,2)):
            l=(left,)*3; r=(right,)*3
            plans.append(f*wait+l+f*width+r+f*(forwards-wait)+r+f*width+l)
    poses=[nominal(p)[-1] for p in plans]
    assert max(np.linalg.norm(q[:4]-poses[0][:4]) for q in poses)<1e-6
    return plans


def collect(split,house):
    dst=RAW/(split+"_"+house)
    dst.mkdir(exist_ok=True,parents=True)
    output=dst/"controls.npz"
    if output.exists():return
    seed=int(hashlib.sha256(("path_feedback:paired:v1:"+house).encode()).hexdigest()[:8],16)%(2**31-1)
    rng=np.random.default_rng(seed)
    sim=simulator(house)
    sim.seed(seed); sim.pathfinder.seed(seed)
    ag=sim.get_agent(0)
    rows=[]
    try:
        sources=choose_sources(sim,inventory(sim,ASSETS),rng,SAMPLES_PER_HOUSE)
        for source_id,(category,sid,p,yaw,inputs) in enumerate(sources):
            src=camera(sim,p,yaw,SOURCE_H,0.)
            rgb=src["rgb"][...,:3].copy(); dep=src["depth"].squeeze().copy()
            state=ag.get_state()
            geo=from_depth(dep)
            l,t,r,b=inputs[0]
            write_rgb(rgb,dst/f"{source_id}_source.jpg")
            write_rgb(rgb[t:b,l:r],dst/f"{source_id}_crop.jpg")
            for n in (2,4,8,12,20):
                for pid,plan in enumerate(options(n)):
                    ag.set_state(state,reset_sensors=False,infer_sensor_states=False)
                    feature,old_safe,x,z,_=features(plan,SOURCE_H,0.,inputs[1],inputs[2],dep)
                    target=p+np.array([x*math.cos(yaw)-z*math.sin(yaw),0,-x*math.sin(yaw)-z*math.cos(yaw)])
                    steps=path_features(plan,geo)
                    path=np.zeros((108,len(NAMES)),np.float16); path[:len(plan)]=steps
                    action=np.zeros(108,np.uint8);action[:len(plan)]=plan
                    blocked=0
                    for k,a in enumerate(plan,1):
                        p0=ag.state.position.copy()
                        contact=ag.act({1:"move_forward",2:"turn_left",3:"turn_right"}[a])
                        displacement=np.linalg.norm((ag.state.position-p0)[[0,2]])
                        if a==1 and (contact or displacement<.23):
                            blocked=k
                            break
                    end=ag.get_state()
                    end_error=np.linalg.norm((end.position-target)[[0,2]])
                    reach=int(blocked==0 and end_error<.14)
                    pixels=-1
                    future=""
                    if reach:
                        obs=sim.get_sensor_observations()
                        pixels=int((obs["ids"]==sid).sum())
                        future=f"{source_id}_{n}_{pid}.jpg"
                        write_rgb(obs["rgb"],dst/future)
                    rows.append(dict(group=f"{house}/{source_id}/d{n}",source_index=source_id,plan_id=pid,
                                     action=action,pose=feature,path=path,category=category,safe_mask=old_safe,
                                     reach=reach,visible=int(pixels>=369) if reach else -1,first_blocked=blocked,
                                     pixels=pixels,end_error_m=end_error,end_x=float(end.position[0]),
                                     end_y=float(end.position[1]),end_z=float(end.position[2]),future_file=future))
    finally:sim.close()
    assert rows
    arr={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    arr.update(house=np.repeat(house,len(rows)),split=np.repeat(split,len(rows)),path_names=np.asarray(NAMES))
    tmp=output.with_suffix(".partial.npz")
    np.savez_compressed(tmp,**arr);tmp.replace(output)
    print("collected",split,house,len(sources),len(rows),int(sum(arr["reach"])),flush=True)


if __name__=="__main__":
    os.umask(0o077)
    p=argparse.ArgumentParser()
    p.add_argument("--worker",type=int,default=0)
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--first",type=int)
    args=p.parse_args()
    jobs=[(split,h) for split,hh in houses().items() for h in hh]
    for split,h in jobs[args.worker::args.workers][:args.first]:
        collect(split,h)
