"""Alternative observations of one cue. All sensor histories and candidate trajectories are physically replayed."""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT/"pose_horizon_20261128"),str(ROOT/"path_feedback_20261128")]
import pose_data
from pose_data import ASSETS,S,SOURCE_H,camera,features,inventory,write_rgb,cue_inputs
from collect_paths import options
from path_geometry import from_depth,path_features,nominal

ACTION={1:"move_forward",2:"turn_left",3:"turn_right"}
from view_geometry import endpoints, map_features

def simulator(house):
    if house != "108736611_177263226":
        return pose_data.simulator(house)
    manifest=ASSETS.parent.parent/"manifests/material_compatibility.json"
    treatment=json.loads(manifest.read_text())
    assert house in treatment["referencing_scenes"]
    compat=Path(treatment["overlay"])
    assert hashlib.sha256((compat/treatment["asset"]).read_bytes()).hexdigest()==treatment["derived_sha256"]
    saved=pose_data.ASSETS
    try:
        pose_data.ASSETS=compat
        return pose_data.simulator(house)
    finally:
        pose_data.ASSETS=saved

def sources(sim,groups,rng,number):
    items=[(cat,sid,c) for cat,obs in sorted(groups.items()) for sid,_,c in obs]
    rng.shuffle(items)
    if not items:return []
    out=[];seen={}
    # Prefer initially small cues for two of every three sources. No future outcomes used.
    for k in range(min(900,len(items)*4)):
        cat,sid,center=items[k%len(items)]
        if seen.get(sid,0)>=2:continue
        small=len(out)%3!=2
        for _ in range(7):
            az=rng.uniform(-math.pi,math.pi);r=rng.uniform(1.6,5.)
            p=np.asarray(sim.pathfinder.snap_point(np.asarray(center+[r*math.sin(az),-1.1,r*math.cos(az)],np.float32)))
            if not np.isfinite(p).all() or not sim.pathfinder.is_navigable(p):continue
            if any(o[1]==sid and np.linalg.norm(o[2]-p)<.6 for o in out):continue
            d=center-p
            if not 1.2<np.linalg.norm(d[[0,2]])<5.2:continue
            yaw=math.atan2(-d[0],-d[2])+rng.uniform(-.30,.30)
            obs=camera(sim,p,yaw,SOURCE_H,0.)
            seg=obs["ids"]==sid
            if not 38<=seg.sum()<=(368 if small else 6500):continue
            cue=cue_inputs(seg,obs["depth"].squeeze())
            if cue is None:continue
            out.append((cat,sid,p.copy(),yaw,cue));seen[sid]=seen.get(sid,0)+1;break
        if len(out)>=number:break
    return out

def collect(split,house):
    folder=HERE/("private_render_blind60" if split=="blind" else "private_render")/(split+"_"+house)
    folder.mkdir(exist_ok=True,parents=True)
    dst=folder/"controls.npz"
    if dst.exists():return
    seed=int(hashlib.sha256(("change_observation:v1:"+house).encode()).hexdigest()[:8],16)%2**31
    rng=np.random.default_rng(seed)
    sim=simulator(house); sim.seed(seed);sim.pathfinder.seed(seed)
    ag=sim.get_agent(0);rows=[]
    try:
        sampled=sources(sim,inventory(sim,ASSETS),rng,60 if split=="blind" else 24)
        for i,(cat,oid,start,yaw,cue) in enumerate(sampled):
            before=camera(sim,start,yaw,SOURCE_H,0.)
            original=ag.get_state()
            ag.act(ACTION[2]);left=sim.get_sensor_observations()
            left_depth=left["depth"].squeeze().copy()
            scan_pixels=int((left["ids"]==oid).sum())
            ag.act(ACTION[3]);source=sim.get_sensor_observations()
            state=ag.get_state()
            assert np.linalg.norm(state.position-original.position)<1e-5 and abs(abs((state.rotation*original.rotation.conjugate()).w)-1)<1e-5
            dep=source["depth"].squeeze().copy();geo=from_depth(dep)
            seg=source["ids"]==oid
            cue=cue_inputs(seg,dep)
            if cue is None:raise ValueError("source cue vanished after yaw restore")
            box,anchor,boxinfo=cue;l,t,r,b=box
            write_rgb(source["rgb"],folder/f"{i}_source.jpg");write_rgb(source["rgb"][t:b,l:r],folder/f"{i}_crop.jpg")
            np.savez_compressed(folder/f"{i}_sensor.npz",depth=dep,left_depth=left_depth,oid=oid,position=start,yaw=yaw,box=box)
            n=(4,8,12)[i%3]
            for kind,vid,choices in endpoints(n,anchor):
                for pid,plan in choices:
                    ag.set_state(state,reset_sensors=False,infer_sensor_states=False)
                    pose,safe,x,z,_=features(plan,SOURCE_H,0.,anchor,boxinfo,dep)
                    target=start+np.array([x*math.cos(yaw)-z*math.sin(yaw),0.,-x*math.sin(yaw)-z*math.cos(yaw)])
                    collision=0
                    for k,a in enumerate(plan,1):
                        old=ag.state.position.copy()
                        contact=ag.act(ACTION[a])
                        if a==1 and (contact or np.linalg.norm((ag.state.position-old)[[0,2]])<.23):
                            collision=k;break
                    err=float(np.linalg.norm((ag.state.position-target)[[0,2]]))
                    reach=not collision and err<.14
                    image="";pixels=-1;future_box=[-1]*4
                    if reach:
                        obs=sim.get_sensor_observations()
                        ys,xs=np.where(obs["ids"]==oid);pixels=len(xs)
                        if pixels:future_box=[xs.min(),ys.min(),xs.max()+1,ys.max()+1]
                        image=f"{i}_{vid}_{pid}.jpg";write_rgb(obs["rgb"],folder/image)
                    acts=np.zeros(108,np.uint8);acts[:len(plan)]=plan
                    path=np.zeros((108,17),np.float16);path[:len(plan)]=path_features(plan,geo)
                    rows.append(dict(group=f"{house}/{i}",source_index=i,kind=kind,view_id=vid,plan_id=pid,
                        action=acts,path=path,pose=np.array(pose,np.float32),map=np.array(map_features(plan,anchor,left_depth),np.float32),
                        budget=n+18,scan_cost=2,category=cat,start_pixels=int(seg.sum()),scan_pixels=scan_pixels,reach=int(reach),
                        visible=int(pixels>=369) if reach else -1,pixels=pixels,first_blocked=collision,end_error_m=err,
                        future_file=image,future_box=np.array(future_box,np.int16),safe_mask=safe))
    finally:sim.close()
    if not rows:raise RuntimeError("no eligible cues for house: "+house)
    tmp=folder/"controls.partial.npz"
    np.savez_compressed(tmp,**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
    tmp.replace(dst);print("collected",split,house,len(sampled),len(rows),flush=True)

if __name__=="__main__":
    os.umask(0o077)
    parser=argparse.ArgumentParser()
    parser.add_argument("--split",choices=["train","val","blind"])
    parser.add_argument("--worker",type=int,default=0)
    parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args()
    older=ROOT/"path_feedback_20261128/natural_design.json"
    design=HERE/"view_design.json"
    d=json.loads((design if design.exists() else older).read_text())["cohorts"]
    if args.split=="blind":assert design.exists()
    jobs=[(split,h) for split in ([args.split] if args.split else ["train","val"]) for h in d[split]]
    for split,house in jobs[args.worker::args.workers]:collect(split,house)
