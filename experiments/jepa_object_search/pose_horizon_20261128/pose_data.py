"""Controlled paired HSSD sensor poses. Truth is used for cue sampling and offline labels."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import habitat_sim as hs
from habitat_sim.utils.common import quat_from_angle_axis as qa
import magnum as mn
import numpy as np
from PIL import Image

from pose_data_contract import FEATURES, EXTRINSICS

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"jepa_reid_v20/collect"))
from collect_views import inventory
ASSETS=Path("/root/autodl-tmp/jepa_vln_data_20261014/raw/hssd-hab")
S=192
SOURCE_H=1.31
GRID=HERE/"private_pose_render"


def houses():
    cohort=ROOT/"jepa_env_v13/simulator"
    c=json.loads((cohort/"cohort.json").read_text())
    return dict(train=json.loads((cohort/"training_cohort.json").read_text())["train"],
                val=c["calibration"],test=c["fresh"])


def simulator(house):
    cfg=hs.SimulatorConfiguration()
    cfg.scene_dataset_config_file=str(ASSETS/"hssd-hab.scene_dataset_config.json")
    cfg.scene_id=house
    cfg.enable_physics=True
    cfg.allow_sliding=False
    ag=hs.agent.AgentConfiguration(height=1.41,radius=.17)
    for key,typ in (("rgb",hs.SensorType.COLOR),("ids",hs.SensorType.SEMANTIC),("depth",hs.SensorType.DEPTH)):
        s=hs.CameraSensorSpec()
        s.uuid,s.sensor_type,s.resolution=key,typ,[S,S]
        s.position,s.hfov=mn.Vector3(0,SOURCE_H,0),mn.Deg(90)
        ag.sensor_specifications.append(s)
    for name,amount in (("move_forward",.25),("turn_left",30.),("turn_right",30.)):
        ag.action_space[name].actuation.amount=amount
    sim=hs.Simulator(hs.Configuration(cfg,[ag]))
    nav=hs.NavMeshSettings()
    nav.set_defaults()
    nav.agent_height,nav.agent_radius,nav.include_static_objects=1.41,.17,True
    if not sim.recompute_navmesh(sim.pathfinder,nav):
        raise ValueError("navmesh failed")
    return sim


def camera(sim,base,yaw,h,pitch):
    st=hs.AgentState()
    st.position=np.asarray(base,np.float32)
    st.rotation=qa(yaw,np.array([0.,1.,0.]))
    ag=sim.get_agent(0)
    ag.set_state(st)
    st=ag.get_state()
    for node in st.sensor_states.values():
        node.position=np.asarray(base)+[0,h,0]
        node.rotation=st.rotation*qa(math.radians(pitch),np.array([1.,0.,0.]))
    ag.set_state(st,reset_sensors=False,infer_sensor_states=False)
    return sim.get_sensor_observations()


def cue_inputs(seg,depth):
    y,x=np.where(seg)
    box=np.array([x.min(),y.min(),x.max()+1,y.max()+1])
    l,t,r,b=box
    # Median centre-window depth, restricted to the supplied rectangle only.
    u,v=(l+r)/2,(t+b)/2
    d=depth[max(0,int(v)-2):int(v)+3,max(0,int(u)-2):int(u)+3]
    d=d[np.isfinite(d)&(d>.05)]
    if not len(d):return None
    z=float(np.median(d))
    anchor=np.array([(u-S/2)*z/(S/2),(S/2-v)*z/(S/2),z])
    return box,anchor,[(l+r)/(2*S),(t+b)/(2*S),(r-l)/S,(b-t)/S,min(10.,z)]


def query_depth(points,depth):
    x,y,z=points.T
    u=(S/2+x*(S/2)/np.maximum(z,.001)).astype(int)
    v=(S/2-y*(S/2)/np.maximum(z,.001)).astype(int)
    inside=(z>.05)&(u>=0)&(v>=0)&(u<S)&(v<S)
    depth_vals=np.zeros_like(z)
    depth_vals[inside]=depth[v[inside],u[inside]]
    supported=inside&np.isfinite(depth_vals)&(depth_vals>.05)
    return supported,(depth_vals+.12<z)&supported


def features(plan,h,pitch,anchor,box_info,depth):
    x,z,yaw=0.,0.,0.
    path=[]
    for a in plan:
        if a==1:
            for t in (.125,.25):
                px=x-t*math.sin(yaw); pz=z+t*math.cos(yaw)
                for y in (-.85,-.40,0.):
                    for side in (-.17,.17):
                        path.append([px+side*math.cos(yaw),y,pz+side*math.sin(yaw)])
            x-=.25*math.sin(yaw); z+=.25*math.cos(yaw)
        else:yaw+=(1 if a==2 else -1)*math.pi/6
    dh=h-SOURCE_H
    rays=np.linspace([x,dh,z],anchor,17)[1:-2]
    seen,occ=query_depth(rays,depth)
    ps,po=query_depth(np.array(path),depth) if path else (np.ones(1,bool),np.zeros(1,bool))
    dx,dy,dz=anchor-[x,dh,z]
    side=dx*math.cos(yaw)+dz*math.sin(yaw)
    fw=-dx*math.sin(yaw)+dz*math.cos(yaw)
    cp,sp=math.cos(math.radians(pitch)),math.sin(math.radians(pitch))
    oy=dy*cp-fw*sp
    oz=dy*sp+fw*cp
    f=[x,z,h,dh,math.sin(yaw),math.cos(yaw),sp,cp]+box_info+[
        np.clip(side/max(.1,oz),-5,5),np.clip(oy/max(.1,oz),-5,5),np.clip(oz,-10,10),
        seen.mean(),occ.sum()/max(1,seen.sum()),ps.mean(),(ps&~po).mean()]
    assert len(f)==len(FEATURES)
    return list(map(float,f)),bool(ps.mean()>=.65 and po.sum()==0),x,z,yaw


def choose_sources(sim,groups,rng,num):
    by_cat={c: list(items) for c,items in groups.items() if items}
    categories=list(by_cat)
    rng.shuffle(categories)
    out=[]
    for index in range(2*len(categories)):
        cat=categories[index%len(categories)]
        objs=by_cat[cat]
        sid,_,center=objs[rng.integers(len(objs))]
        if any(v[1]==sid for v in out):continue
        for _ in range(16):
            ang=rng.uniform(-math.pi,math.pi); rad=rng.uniform(1.5,4.)
            point=center+[rad*math.sin(ang),-1.1,rad*math.cos(ang)]
            p=np.asarray(sim.pathfinder.snap_point(np.float32(point)),np.float32)
            if not np.isfinite(p).all() or not sim.pathfinder.is_navigable(p):continue
            d=center-p
            if not 1.<np.linalg.norm(d[[0,2]])<4.5:continue
            yaw=math.atan2(-d[0],-d[2])+rng.uniform(-.40,.40)
            o=camera(sim,p,yaw,SOURCE_H,0.)
            seg=o["ids"]==sid
            if seg.sum()<40 or seg.sum()>.31*S*S:continue
            inputs=cue_inputs(seg,np.asarray(o["depth"]).squeeze())
            if inputs:
                out.append((cat,sid,p.copy(),yaw,inputs))
                break
        if len(out)>=num:break
    return out


def write_rgb(img,path):
    Image.fromarray(img[...,:3]).save(path,quality=92)


def check_occlusion(points,x,z,theta,h,pitch,obs,sid):
    # Label only: reproject previously seen surface points of THIS instance, then
    # check that a different surface now intercepts their rays at a smaller depth.
    dx,dy,dz=(points-[x,h-SOURCE_H,z]).T
    sx=dx*math.cos(theta)+dz*math.sin(theta)
    fw=-dx*math.sin(theta)+dz*math.cos(theta)
    cp,sp=math.cos(math.radians(pitch)),math.sin(math.radians(pitch))
    sy,sz=dy*cp-fw*sp,dy*sp+fw*cp
    u=S/2+(S/2)*sx/np.maximum(.02,sz)
    v=S/2-(S/2)*sy/np.maximum(.02,sz)
    ok=(sz>.05)&(u>=0)&(u<S)&(v>=0)&(v<S)
    ui,vi=u[ok].astype(int),v[ok].astype(int)
    d=obs["depth"].squeeze()[vi,ui]
    met=(d>.05)&(d<sz[ok]-.15)&(obs["ids"][vi,ui]!=sid)
    return int(ok.sum()),int(met.sum())


def collect(split,house):
    dest=GRID/(split+"_"+house)
    dest.mkdir(parents=True,exist_ok=True)
    target=dest/"samples.json"
    if target.exists():return
    rng=np.random.default_rng(int(hashlib.sha256(house.encode()).hexdigest()[:8],16))
    sim=simulator(house)
    sim.seed(821)
    rows=[]
    try:
        groups=inventory(sim,ASSETS)
        sources=choose_sources(sim,groups,rng,10)
        for j,(cat,sid,p0,yaw,inputs) in enumerate(sources):
            source=camera(sim,p0,yaw,SOURCE_H,0)
            rgb=source["rgb"].copy()
            dep=source["depth"].squeeze().copy()
            yy,xx=np.where((source["ids"]==sid)&(dep>.05)&np.isfinite(dep))
            ii=np.linspace(0,len(xx)-1,min(32,len(xx)),dtype=int)
            yy,xx=yy[ii],xx[ii]
            sd=dep[yy,xx]
            vpoints=np.stack(((xx+.5-S/2)*sd/(S/2),(S/2-yy-.5)*sd/(S/2),sd),axis=1)
            box,anchor,box_info=inputs
            l,t,r,b=box
            write_rgb(rgb,dest/f"{j}_src.jpg")
            write_rgb(rgb[t:b,l:r],dest/f"{j}_cue.jpg")
            # Candidate plans are sampled independent of any future labels.
            fw=(1,)*int(rng.choice([2,4,6,8]))
            plans=((2,3),(2,),(3,),fw,(2,)+fw+(3,),(3,)+fw+(2,),
                   (2,2)+fw+(3,),(3,3)+fw+(2,),(1,)*10)
            for pid,plan in enumerate(plans):
                camera(sim,p0,yaw,SOURCE_H,0)
                _,_,x,z,theta=features(plan,SOURCE_H,0.,anchor,box_info,dep)
                ideal=p0+np.array([x*math.cos(yaw)-z*math.sin(yaw),0,-x*math.sin(yaw)-z*math.cos(yaw)])
                slipped=False
                for a in plan:
                    prior=sim.get_agent(0).state.position.copy()
                    sim.step({1:"move_forward",2:"turn_left",3:"turn_right"}[a])
                    end=sim.get_agent(0).state.position
                    if a==1 and np.linalg.norm((end-prior)[[0,2]])<.23:
                        slipped=True
                        break
                end=sim.get_agent(0).state.position
                right_spot=np.linalg.norm((ideal-end)[[0,2]])<.14
                reach=not slipped and right_spot
                p=np.asarray(sim.pathfinder.snap_point(np.float32(ideal)),np.float32)
                valid=np.isfinite(p).all() and np.linalg.norm((p-ideal)[[0,2]])<.11 and abs(p[1]-p0[1])<.4
                if reach:
                    p=end.copy()
                    valid=True
                for ei,(h,pitch) in enumerate(EXTRINSICS):
                    f,safe,*_=features(plan,h,pitch,anchor,box_info,dep)
                    future=""
                    pix=-1
                    in_view=occluded=-1
                    if valid:
                        obs=camera(sim,p,yaw+theta,h,pitch)
                        pix=int((obs["ids"]==sid).sum())
                        in_view,occluded=check_occlusion(vpoints,x,z,theta,h,pitch,obs,sid)
                        future=f"{j}_{pid}_{ei}.jpg"
                        write_rgb(obs["rgb"],dest/future)
                    rows.append(dict(group=house+"/"+str(j),plan_id=pid,category=cat,
                        source=f"{j}_src.jpg",cue=f"{j}_cue.jpg",future=future,
                        pose=f,action=plan,reach=int(reach),at_position=int(right_spot),
                        visible=int(pix>=math.ceil(.01*S*S)) if valid else -1,
                        pixels=pix,safe_mask=safe,height=h,pitch_deg=pitch,native_mask=ei==0,
                        source_point_count=len(vpoints),in_fov=in_view,intercepted=occluded))
    finally:sim.close()
    tmp=dest/"rows.tmp"
    tmp.write_text(json.dumps(rows,allow_nan=False))
    tmp.replace(target)
    print(split,house,len(rows),sum(r["reach"] for r in rows),sum(r["visible"]==1 for r in rows),flush=True)


if __name__=="__main__":
    os.umask(0o077)
    parser=argparse.ArgumentParser()
    parser.add_argument("--worker",type=int,default=0)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--one",action="store_true")
    args=parser.parse_args()
    tasks=[(s,h) for s,hh in houses().items() for h in hh]
    for split,house in tasks[args.worker::args.workers][:1 if args.one else None]:
        collect(split,house)
