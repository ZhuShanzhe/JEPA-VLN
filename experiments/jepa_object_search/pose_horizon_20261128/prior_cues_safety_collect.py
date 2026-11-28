"""Project the same prospective body-clearance points into the original three measured depth views."""
import math
import os
from pathlib import Path
import numpy as np
from prior_cues_collect import (HERE,MV,FAMILY,PLANS,environment,physical,ScanRecorder,
                               case_indices,read)
from pose_data import query_depth


def points(plan):
    pts=[]
    x=z=yaw=0.
    for a in plan:
        if a==1:
            for dist in (.125,.25):
                px=x-dist*math.sin(yaw)
                pz=z+dist*math.cos(yaw)
                for y in (-.85,-.4,0.):
                    for side in (-.17,.17):
                        pts.append((px+side*math.cos(yaw),y,pz+side*math.sin(yaw)))
            x-=.25*math.sin(yaw)
            z+=.25*math.cos(yaw)
        else:yaw+=(1 if a==2 else -1)*math.pi/6
    return np.array(pts)


def reframe(pts,src,other):
    # Habitat policy ground coordinates (world X,-world Z). Y is camera-relative and all 3 cameras are level.
    right=lambda a: np.array([math.sin(a),-math.cos(a)])
    forward=lambda a: np.array([math.cos(a),math.sin(a)])
    world=np.asarray(src[:2])+np.outer(pts[:,0],right(src[2]))+np.outer(pts[:,2],forward(src[2]))
    rel=world-np.asarray(other[:2])
    return np.stack((rel@right(other[2]),pts[:,1],rel@forward(other[2])),axis=1)


def coverage(geometry,source,plan):
    pts=points(plan)
    if not len(pts):return np.array([1.,1.,0.,1.])
    seen=np.zeros(len(pts),bool)
    hazard=seen.copy()
    for i,(pose,depth) in enumerate(geometry):
        coords=pts if i==source else reframe(pts,geometry[source][0],pose)
        is_seen,occluded=query_depth(coords,depth)
        seen|=is_seen
        hazard|=occluded
    return np.asarray([seen.mean(),(seen&~hazard).mean(),hazard.mean(),float(seen.mean()>=.65 and not np.any(hazard))])


def extract(i,spec,accepted):
    wanted,trace=case_indices(i,accepted)
    if not wanted:return 0
    ep,sim=environment(spec)
    try:
        state=physical.hs.AgentState()
        state.position=np.float32(ep["start_position"])
        state.rotation=physical.quat_from_coeffs(ep["start_rotation"])
        sim.get_agent(0).set_state(state)
        scan=ScanRecorder(sim)
        scan.get_sensor_observations()
        for step,t in enumerate(trace[1:max(wanted)+1],1):
            scan.step(t["action"])
            if step not in wanted:continue
            k,e,_=wanted[step]
            frames=list(scan.frames)
            geo,rec=scan.packet([frames[j]["rgb"] for j in (1,0,3)],frames[0]["depth"],physical.policy_pose(sim))
            assert all(a["rgb_sha256"]==b["rgb_sha256"] and a["depth_sha256"]==b["depth_sha256"] and a["step"]==b["step"]
                       for a,b in zip(rec,e["physical_scan"]))
            src=e["cue_view_index"]
            name=f"{i}_{k}.npz"
            with np.load(HERE/"private_prior_cues"/name) as old:
                single=np.stack([coverage([geo[src]],0,p) for p in PLANS])
                assert np.allclose(single[:,:2],old["pose"][:,-2:],atol=1e-6)
                assert np.array_equal(single[:,3].astype(bool),old["safe"])
                three=np.stack([coverage(geo,src,p) for p in PLANS])
                assert np.all(three[:,0]>=single[:,0]) and np.all(three[:,2]>=single[:,2])
            np.savez_compressed(HERE/"private_prior_safety"/name,single=single,three=three)
            print("verified",i,k,flush=True)
    finally:sim.close()
    return len(wanted)


if __name__=="__main__":
    os.umask(0o077)
    (HERE/"private_prior_safety").mkdir(exist_ok=True)
    accepted=read(MV/"interface_review.json")
    assert sum(extract(i,s,accepted) for i,s in enumerate(read(FAMILY/"panel.json")))==4
