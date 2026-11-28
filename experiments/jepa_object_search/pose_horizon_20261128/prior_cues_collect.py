"""Replay all previously verified side detections without a strict local site. Original cameras only."""
from collections import deque
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
from quaternion import rotate_vectors

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
MV=ROOT/"multiview_nav_20261128"
sys.path.insert(0,str(MV))
from run_multiview import ScanRecorder, environment, physical, FAMILY, read, digest
from review_interface import configured_sim, instance_lut, box_match, DATA, _load_goal_sets, load_scene_instances
from pose_data import features, SOURCE_H, S

F=(1,)*4
PLANS=((2,3),(2,),(3,),F,(2,)+F+(3,),(3,)+F+(2,),(2,2)+F+(3,),(3,3)+F+(2,),(1,)*10)
ACT={1:"move_forward",2:"turn_left",3:"turn_right"}
OUT=HERE/"private_prior_cues"


def case_indices(house,accepted):
    gold=accepted["houses"][house]["selected_cues"]
    events=MV/"private_events"/f"{house}.jsonl"
    poses=MV/"private_events"/f"{house}.poses.jsonl"
    signed=read(MV/"runs"/f"{house}_multiview.json")
    assert digest(events)==signed["private_events_sha256"] and digest(poses)==signed["private_trace_sha256"]
    selected=[row for row in map(json.loads,events.read_text().splitlines()) if row.get("cue_selected")]
    assert len(selected)==len(gold)
    wanted={e["step"]:(k,e,g) for k,(e,g) in enumerate(zip(selected,gold))
            if g["valid_box"] and g["anchored"] and not g["site_offered"]}
    return wanted,list(map(json.loads,poses.read_text().splitlines()))


def observed_geometry(box,depth):
    b=np.asarray(box,dtype=np.float64)*S
    l,t,r,d=b
    u,v=(l+r)/2,(t+d)/2
    window=depth[max(0,int(v)-2):int(v)+3,max(0,int(u)-2):int(u)+3]
    valid=window[(window>.05)&np.isfinite(window)]
    if not len(valid):raise ValueError("Selected box has no measured center-window depth")
    z=float(np.median(valid))
    return (np.array([(u-S/2)*z/(S/2),(S/2-v)*z/(S/2),z]),
            [u/S,v/S,(r-l)/S,(d-t)/S,min(10.,z)])


def motion(sim,source,rgb,depth,box):
    anchor,geom=observed_geometry(box,depth)
    records=[]
    for plan in PLANS:
        ag=sim.get_agent(0)
        ag.set_state(source)
        now=sim.get_sensor_observations()
        rgb_delta=np.abs(rgb.astype(np.int16)-now["rgb"][:,:,:3].astype(np.int16))
        delta=np.abs(now["depth"].astype(np.float64)-depth.astype(np.float64))
        assert np.mean(rgb_delta)<.12 and np.mean(delta)<.0005,(np.mean(rgb_delta),np.mean(delta))
        p,safe,x,z,_=features(plan,SOURCE_H,0.,anchor,geom,depth)
        ideal=source.position+rotate_vectors(source.rotation,[x,0.,-z])
        stalled=False
        for action in plan:
            before=ag.state.position.copy()
            sim.step(ACT[action])
            if action==1 and np.linalg.norm((ag.state.position-before)[[0,2]])<.23:
                stalled=True
                break
        reached=(not stalled and np.linalg.norm((ag.state.position-ideal)[[0,2]])<.14)
        records.append(dict(features=p,safe=safe,end=ag.state,reach=reached,
                        rgb_mean=float(np.mean(rgb_delta)),depth_mean=float(np.mean(delta))))
    return records


def cases(house,spec,accepted):
    wanted,trace=case_indices(house,accepted)
    if not wanted:return 0
    ep,sim=environment(spec)
    found=[]
    try:
        agent=sim.get_agent(0)
        st=physical.hs.AgentState()
        st.position=np.float32(ep["start_position"])
        st.rotation=physical.quat_from_coeffs(ep["start_rotation"])
        agent.set_state(st)
        scan=ScanRecorder(sim)
        scan.get_sensor_observations()
        states=deque([agent.state],maxlen=5)
        for step,t in enumerate(trace[1:max(wanted)+1],1):
            scan.step(t["action"])
            cur=agent.state
            states.append(cur)
            assert np.allclose(cur.position,t["xyz"],atol=1e-5,rtol=0)
            if step not in wanted:continue
            index,event,gold=wanted[step]
            fs=list(scan.frames)
            _,receipts=scan.packet([fs[j]["rgb"] for j in (1,0,3)],fs[0]["depth"],physical.policy_pose(sim))
            assert all(r["rgb_sha256"]==g["rgb_sha256"] and r["depth_sha256"]==g["depth_sha256"] and r["step"]==g["step"]
                       and np.allclose(r["pose"],g["pose"],rtol=0,atol=1e-6) for r,g in zip(receipts,event["physical_scan"]))
            vi=event["cue_view_index"]
            assert vi in (0,2)
            j=(1,0,3)[vi]
            found.append((index,step,event["cue_bbox_norm"],gold,states[j],fs[j]["rgb"],fs[j]["depth"]))
        motions=[motion(sim,st,rgb,dep,box) for _,_,box,_,st,rgb,dep in found]
    finally:sim.close()
    semantic=configured_sim(ep)
    try:
        lut=instance_lut(semantic,"hssd",DATA/"vocab")
        ids=_load_goal_sets(DATA)["scoring_class_ids"]["hssdov"][ep["object_category"]]
        targets={r["semantic_id"] for r in load_scene_instances(DATA/"protocol/instance_viewpoints/hssd","val",ep["scene_id"])
                 if r["class_id"] in ids and r["view_points"]}
        assert all(lut[s]-1 in ids for s in targets)
        def mask(st):
            semantic.get_agent(0).set_state(st)
            return semantic.get_sensor_observations()["instance_mask"]
        for (index,step,box,gold,source,rgb,depth),rows in zip(found,motions):
            m=mask(source)
            sid,iou,_=box_match(np.where(np.isin(m,list(targets)),m,0),box)
            assert sid and abs(iou-gold["iou_same_instance"])<.0002
            l,t=np.clip(np.floor(np.asarray(box[:2])*S).astype(int),0,S-1)
            r,b=np.clip(np.ceil(np.asarray(box[2:])*S).astype(int),1,S)
            p=np.array([int((mask(row["end"])==sid).sum()) for row in rows])
            assert p[0]>=0
            dest=OUT/f"{house}_{index}.npz"
            plans=np.zeros((len(rows),12),np.int8)
            for j,plan in enumerate(PLANS):plans[j,:len(plan)]=plan
            np.savez_compressed(dest,rgb=rgb,depth=depth,bbox=box,crop=rgb[t:b,l:r],action=plans,
                                pose=np.array([r["features"] for r in rows],np.float32),
                                safe=np.array([r["safe"] for r in rows],bool),
                                reach=np.array([r["reach"] for r in rows],bool),pixels=p,
                                source_pixels=int((m==sid).sum()),source_sha_exact=True,
                                rgb_reset_mae=max(r["rgb_mean"] for r in rows),
                                depth_reset_mae=max(r["depth_mean"] for r in rows))
            print("replayed",house,index,step,"same-instance pixels",p.tolist(),flush=True)
    finally:semantic.close()
    return len(found)


if __name__=="__main__":
    os.umask(0o077)
    OUT.mkdir(exist_ok=True)
    accepted=read(MV/"interface_review.json")
    assert sum(cases(i,s,accepted) for i,s in enumerate(read(FAMILY/"panel.json")))==4
