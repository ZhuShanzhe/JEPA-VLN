"""Physical same-waypoint forks on four archived detector cues, including optional charged yaw observations."""
from collections import deque
import os
from pathlib import Path
import sys
import numpy as np
from quaternion import rotate_vectors

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/"pose_horizon_20261128"
sys.path.insert(0,str(PRIOR))
from prior_cues_collect import (case_indices,read,MV,FAMILY,environment,physical,ScanRecorder,configured_sim,
    instance_lut,DATA,_load_goal_sets,load_scene_instances,box_match,ACT,observed_geometry,features)
from collect_paths import options
from path_geometry import nominal


def collect(i,spec,accepted):
    wanted,trace=case_indices(i,accepted)
    if not wanted:return
    ep,sim=environment(spec)
    sem=configured_sim(ep)
    try:
        instance_lut(sem,"hssd",DATA/"vocab")
        ids=_load_goal_sets(DATA)["scoring_class_ids"]["hssdov"][ep["object_category"]]
        targets={r["semantic_id"] for r in load_scene_instances(DATA/"protocol/instance_viewpoints/hssd","val",ep["scene_id"])
                 if r["class_id"] in ids and r["view_points"]}
        a=sim.get_agent(0)
        state=physical.hs.AgentState()
        state.position=np.float32(ep["start_position"])
        state.rotation=physical.quat_from_coeffs(ep["start_rotation"])
        a.set_state(state)
        scan=ScanRecorder(sim); scan.get_sensor_observations()
        queue=deque([a.state],maxlen=5)
        sources=[]
        for step,t in enumerate(trace[1:max(wanted)+1],1):
            scan.step(t["action"])
            queue.append(a.state)
            if step in wanted:
                index,event,_=wanted[step]
                j=(1,0,3)[event["cue_view_index"]]
                sources.append((index,queue[j],event))
        for index,initial,event in sources:
            filename=f"{i}_{index}.npz"
            with np.load(PRIOR/"private_prior_cues"/filename) as old:
                old_rgb=old["rgb"];old_depth=old["depth"];crop=old["crop"]
            a.set_state(initial)
            obs=sim.get_sensor_observations()
            assert np.array_equal(obs["rgb"][...,:3],old_rgb) and np.array_equal(obs["depth"],old_depth)
            dep=[obs["depth"].squeeze().copy()]
            # These hypothetical scans each originate at the original cue position; advancing from L to R costs two turns.
            for side in (2,3):
                a.set_state(initial);dep.append(sim.step(ACT[side])["depth"].squeeze().copy())
            sem.get_agent(0).set_state(initial)
            mask=sem.get_sensor_observations()["instance_mask"]
            sid,iou,_=box_match(np.where(np.isin(mask,list(targets)),mask,0),event["cue_bbox_norm"])
            assert sid and iou>=.25
            plans=options(4)
            anchor,geom=observed_geometry(event["cue_bbox_norm"],old_depth.squeeze())
            pose=np.array([features(plan,1.31,0.,anchor,geom,dep[0])[0] for plan in plans],np.float32)
            goal=initial.position+rotate_vectors(initial.rotation,[0,0,-1.])
            labels=[]
            for view,scan_plan in enumerate(((),(2,),(2,3,3))):
                undo=((),(3,),(2,))[view]
                vlabels=[]
                for p in plans:
                    a.set_state(initial)
                    for act in scan_plan:sim.step(ACT[act])
                    reached=False;blocked=False;issued=len(scan_plan)
                    for act in undo+p:
                        before=a.state.position.copy()
                        sim.step(ACT[act]);issued+=1
                        if act==1 and np.linalg.norm((a.state.position-before)[[0,2]])<.23:
                            blocked=True;break
                    reached=not blocked and np.linalg.norm((a.state.position-goal)[[0,2]])<.14
                    sem.get_agent(0).set_state(a.state)
                    pix=np.count_nonzero(sem.get_sensor_observations()["instance_mask"]==sid)
                    vlabels.append((reached,blocked,pix,issued))
                labels.append(vlabels)
            acts=np.zeros((len(plans),108),np.int8)
            for j,p in enumerate(plans):acts[j,:len(p)]=p
            np.savez_compressed(HERE/"private_history"/filename,rgb=old_rgb,crop=crop,
                depth=np.array(dep),action=acts,pose=pose,labels=np.asarray(labels,dtype=int))
    finally:
        sem.close();sim.close()


if __name__=="__main__":
    os.umask(0o077)
    (HERE/"private_history").mkdir(exist_ok=True)
    accepted=read(MV/"interface_review.json")
    for i,spec in enumerate(read(FAMILY/"panel.json")):collect(i,spec,accepted)
    assert len(list((HERE/"private_history").glob("*.npz")))==4
    print("Verified four real detector sources and re-executed 84 counterfactual same-waypoint paths")
