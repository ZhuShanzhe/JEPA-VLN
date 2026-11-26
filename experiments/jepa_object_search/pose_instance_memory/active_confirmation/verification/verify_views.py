"""Reproduce the recorded physical start. Semantic pixels are only for post-trial scoring."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import math
import os
import sys
import numpy as np

P = Path(os.environ["NAV_ROOT"])
for relative in ("jepa_exec_v14","jepa_completion_v17/physical","jepa_reid_v21/detector",
                 "jepa_reid_v23/report","jepa_report_v19","jepa_reid_v24/view"):
    sys.path.insert(0, str(P/relative))
from verify_habitat import replay_until, state, deny_network, difference
sys.addaudithook(deny_network)
from check_routes import semantic_scene
from evaluation.main.evaluator import _env_action
from agents import navigator
from appearance import build_reference, match_reference
from report_evidence import current_support, eligible_boxes

def load(p):
    return json.loads(p.read_bytes())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def rgb(frame):
    return np.asarray(frame["rgb"])[...,:3].copy()

def case():
    root=P/"mapping_runtime/pilots/jepa_reid_v23_first2"
    row=load(root/"panel.json")["rows"][0]
    receipt=load(root/"comparison/accepted/second"/(row["episode_id"]+".json"))
    path=Path(receipt["result"])
    assert sha(path)==receipt["result_sha256"]
    doc=load(path)
    ep,=doc["episodes"]
    ws=next(json.loads(raw)["world_state"] for raw in (path.parent/"agent/decision_trace.jsonl").read_bytes().splitlines()
            if json.loads(raw).get("step")==74)
    box=ws["local_rgb_proposals"]["boxes"][0]
    return doc,ep,box,sha(path)

def physical(env,action):
    old=state(env)
    frame=env.step(_env_action(int(action)))
    translation,turn=difference(old,state(env))
    return frame,dict(action=int(action),translation_m=round(translation,4),turn_deg=round(turn,2),
                      collided=bool(env.sim.previous_step_collided) if action==1 else False)

def input_setup(env,item,data,ep,box):
    frame=replay_until(env,item,data,ep["paper"]["steps"],74)
    origin=state(env)[0]
    def pose():
        from habitat.utils.geometry_utils import quaternion_rotate_vector
        here=env.sim.get_agent_state()
        f=quaternion_rotate_vector(here.rotation,np.array([0.,0.,-1.]))
        x,y=here.position[0]-origin[0],origin[2]-here.position[2]
        return (float(x),float(y),math.atan2(-f[2],f[0]))
    saved=pose()
    ref=build_reference(rgb(frame),box)
    assert ref is not None
    poses=[pose()]
    masks=[np.asarray(frame["semantic"]).copy()]
    pre=[]
    for act in (2,2,2,2,1,1,1):
        frame,row=physical(env,act)
        pre.append(row)
        poses.append(pose())
        masks.append(np.asarray(frame["semantic"]).copy())
    return frame,pose,saved,ref,poses,masks,pre

def verify_pixels(item, first_mask, original_box, samples):
    x,y,u,v=(np.array(original_box)/4).round().astype(int)
    ids=set(map(int,item.target_object_ids))
    hits={k:int(np.count_nonzero(first_mask[y:v,x:u]==k)) for k in ids}
    first_id=max(hits,key=hits.get)
    if hits[first_id]==0:
        return dict(reference_task_target=False,source_max_task_pixels_inside_box=0)
    return dict(reference_task_target=True,source_task_pixels_inside_box=hits[first_id],
                source_box_area=int((u-x)*(v-y)),pixels_each_view=[int(np.count_nonzero(m==first_id)) for m in samples])

def probe():
    doc,ep,box,digest=case()
    env,item,data=semantic_scene(doc)
    try:
        frame,pose,saved,ref,poses,masks,pre=input_setup(env,item,data,ep,box)
        print(json.dumps(dict(initial_pose=saved,final_pose=pose(),pre=pre,observation_fields=list(frame),
                              audit=verify_pixels(item,masks[0],box,masks))))
    finally:
        env.close()

if __name__=="__main__" and "--probe" in sys.argv:
    probe()
def inspect_sources():
    doc,ep,_,digest=case()
    root=P/"mapping_runtime/pilots/jepa_reid_v23_first2"
    eid=ep["episode_id"]
    trace=Path(load(root/"comparison/accepted/second"/(eid+".json"))["result"]).parent/"agent/decision_trace.jsonl"
    env,item,data=semantic_scene(doc)
    try:
        print("sensor",env.sim.habitat_config.agents.main_agent.sim_sensors.depth)
        for raw in trace.read_bytes().splitlines():
            d=json.loads(raw); step=d["step"]; props=d["world_state"].get("local_rgb_proposals") or {}; boxes=props.get("boxes",[])
            if not boxes or step not in (12,74,91): continue
            frame=replay_until(env,item,data,ep["paper"]["steps"],step)
            mask=np.asarray(frame["semantic"]); print("source", step, [verify_pixels(item,mask,bb,[mask]) for bb in boxes])
    finally:
        env.close()

if __name__=="__main__" and "--source-audit" in sys.argv:
    inspect_sources()
def execute(env,item,data,ep,box,slot,worker,empty_detection=False):
    from active_agent import ActiveFrontier
    from benchmark_api import Action
    # Range and orientation of the supplied nearby registration are inputs in the
    # controlled test. They do not use target IDs, simulator object centres or navmesh.
    frame,pose,anchor,reference,positions,masks,pre=input_setup(env,item,data,ep,box)
    yaw=anchor[2]
    # Registered camera is origin, its positive horizontal heading is (cos,sin).
    camera=np.eye(4)
    camera[:3,:3]=np.array([[-math.sin(yaw),0,math.cos(yaw)],
                              [math.cos(yaw),0,math.sin(yaw)],[0,1,0]])
    bw=float(env.sim.habitat_config.agents.main_agent.sim_sensors.rgb.width)
    hfov=math.radians(float(env.sim.habitat_config.agents.main_agent.sim_sensors.rgb.hfov))
    bearing=math.atan((1.-(box[0]+box[2])/bw)*math.tan(hfov/2))
    # This normalized near-range input is held fixed across boxes and controls.
    # The original 74th-step V23 estimate is not being evaluated here.
    r=.60
    point=(r*math.cos(yaw+bearing),r*math.sin(yaw+bearing))
    node=SimpleNamespace(iid=1,point=(*point,0.),observation_ids=[1],reported=False)
    res=.05; low=np.array([-2.5,-2.5]); xx,yy=np.meshgrid(low[0]+(np.arange(100)+.5)*res,
                                                            low[1]+(np.arange(100)+.5)*res)
    # Only a corridor surrounding the already traversed pre-input positions.
    free=np.zeros(xx.shape,bool)
    for a,b in zip(positions[:-1],positions[1:]):
        for t in np.linspace(0,1,11):
            cx,cy=(1-t)*np.array(a[:2])+t*np.array(b[:2])
            free |= (xx-cx)**2+(yy-cy)**2 <= .23**2
    grid=navigator.OccupancyGrid(res,low,free,np.zeros_like(free))
    ag=ActiveFrontier.__new__(ActiveFrontier)
    ag._confirm_view_state=None; ag._view_history={}; ag._view_last_reply=None
    ag._pending_report=None; ag._goal_images=None; ag._instance_goal_index={}; ag.report_near_dist_m=.8
    ag._last_motion_failed=False; ag._estimated_current_pose=pose; ag._metric_scale_value=lambda:1.
    ag._references=lambda n:[(74,reference)]; ag._frontier_grid=grid; ag.align_R=np.eye(3)
    ag.client=SimpleNamespace(get_frame_pose=lambda fid:camera if fid==74 else None)
    ag.instance_store=SimpleNamespace(nodes=[node],get=lambda i:node if i==node.iid else None); ag._collect_views=lambda:None
    ag._known_owner=lambda n:None
    ag._live_verdict_cache={}; ag._verdict_step=None; ag._match_cache_step=None; ag._match_cache={}
    ag._last_feed_info={}; ag._capture_report_view=lambda:None
    ag._category=lambda _: "distinct clock objects"
    events=[]; detector_counts=[]
    ag._note=lambda kind,iid,**values:events.append(dict(event=kind,**values))
    def detections(key,image,category):
        rows=[] if empty_detection else worker.detect(image,category)
        detector_counts.append(len(rows))
        return rows
    ag._boxes=detections
    ag._requested=set(); ag._report_feedback={}; ag._stamp_pending_outcome=lambda *args:None
    step=81; previous=1
    motions=[]; after=[]; emitted=False
    for _ in range(14):
        obs=SimpleNamespace(step_count=step,previous_action=previous,rgb=rgb(frame))
        ag.now=step
        ag._last_observation=obs
        if not motions:
            # Same normal entry point reached by a preserved VLM report request.
            action=ag._report_found(node.iid)
        elif ag._confirm_view_state and ag._confirm_view_state["cursor"] < len(ag._confirm_view_state["sequence"]):
            action=ag._auto_report(obs)
        else:
            reply=ag._confirm_view_step(node,obs)
            action=reply[0] if reply is not None else None
        if action is None:
            break
        if action==int(Action.TARGET_FOUND):
            assert ag._pending_report is not None
            emitted=True
            break
        assert action in (1,2,3)
        frame,m=physical(env,action)
        motions.append(m)
        after.append(np.asarray(frame["semantic"]).copy())
        ag._last_motion_failed=(action==1 and m["translation_m"]<.005)
        previous=action; step+=1
    # The frozen native judge gets privileged data only after the last action.
    from evaluation.main.evaluator import _near_target_ids, _classify_target_claims, _update_claim_quotas
    pos=tuple(env.sim.get_agent_state().position)
    near=_near_target_ids(pos,item.target_object_ids,item.target_positions,threshold=.25,
                          pathfinder=env.sim.pathfinder,target_view_points=item.view_points)
    x,y,u,v=(np.asarray(box)/4).round().astype(int)
    ids=[int(k) for k in item.target_object_ids]
    source_hits=[int(np.count_nonzero(masks[0][y:v,x:u]==k)) for k in ids]
    source_id=ids[int(np.argmax(source_hits))] if max(source_hits)>0 else None
    native=dict(targets_in_range=len(near),reference_in_range=(source_id in set(map(int,near))) if source_id is not None else None)
    assert item.target_mode=="many" and not ep["paper"]["steps"][74]["reached_target_ids_before"]
    assert not [e for e in ep["report_events"] if e["step"]<74 and e.get("near_targets")]
    native["credited_new_instances"]=0
    if emitted:
        quotas={int(k):v for k,v in ep["paper"]["steps"][74]["claim_quotas_before"].items()}
        _update_claim_quotas(quotas,near)
        credited,within,over=_classify_target_claims(near,{},quotas)
        native["credited_new_instances"]=len(credited)
        native["source_region_matches_credited_instance"]=source_id in credited if source_id is not None else False
        native["duplicate_or_over_quota"]=bool(over)
    return dict(entry_point="_report_found then _auto_report",native_viewpoint_audit=native,box_slot=slot,empty_live_detections=empty_detection,source_step=74,
                input_range_m=r,ingress_actions=pre,controller_motion=motions,
                live_detector_count=detector_counts,dispatched=emitted,events=events,
                after_action_audit=verify_pixels(item,masks[0],box,[masks[0],masks[-1],*after]))

def full():
    from proposal_worker import ProposalWorker
    doc,ep,_,digest=case()
    root=P/"mapping_runtime/pilots/jepa_reid_v23_first2"
    path=Path(load(root/"comparison/accepted/second"/(ep["episode_id"]+".json"))["result"])
    trace=path.parent/"agent/decision_trace.jsonl"
    rec=next(json.loads(raw) for raw in trace.read_bytes().splitlines() if json.loads(raw)["step"]==74)
    boxes=rec["world_state"]["local_rgb_proposals"]["boxes"]
    assert len(boxes)==2
    env,item,data=semantic_scene(doc)
    detector=ProposalWorker(P/"b_verification/env/bin/python",P,"cuda")
    try:
        runs=[execute(env,item,data,ep,box,i,detector) for i,box in enumerate(boxes)]
        # Counterfactual missing detector response from the same physical frames.
        runs.append(execute(env,item,data,ep,boxes[1],1,detector,empty_detection=True))
    finally:
        detector.close()
        env.close()
    sources=[P/"jepa_reid_v24/active_agent.py",P/"jepa_reid_v24/view/confirm_view.py",
             P/"jepa_reid_v24/evidence/candidate_evidence.py",P/"jepa_report_v19/appearance.py",
             P/"jepa_reid_v23/report/report_recovery.py",P/"jepa_reid_v23/report/report_evidence.py",
             Path(__file__).resolve()]
    result=dict(archived_result_sha256=digest,runs=runs,paid_requests=0,
                sources={str(path):sha(path) for path in sources})
    (P/"jepa_reid_v24/sim/runs.json").write_text(json.dumps(result,indent=2))
    print("AUDIT",json.dumps([{k:r[k] for k in ("box_slot","after_action_audit","live_detector_count","dispatched")} for r in runs]))

if __name__=="__main__" and "--full" in sys.argv:
    full()
def anchor_check():
    from proposal_worker import ProposalWorker
    doc,ep,_,digest=case()
    root=P/"mapping_runtime/pilots/jepa_reid_v23_first2"; eid=ep["episode_id"]
    path=Path(load(root/"comparison/accepted/second"/(eid+".json"))["result"])
    objects=[(r["step"],r["world_state"]["local_rgb_proposals"]["boxes"]) for raw in (path.parent/"agent/decision_trace.jsonl").read_bytes().splitlines() if (r:=json.loads(raw))["step"] in (12,74,91) and r["world_state"].get("local_rgb_proposals",{}).get("boxes",[])]
    env,item,data=semantic_scene(doc); detector=ProposalWorker(P/"b_verification/env/bin/python",P,"cuda"); records=[]
    try:
        for step,boxes in objects:
            frame=replay_until(env,item,data,ep["paper"]["steps"],step); im=rgb(frame)
            current=eligible_boxes(detector.detect(im,"distinct clock objects"),im.shape)
            masks=np.asarray(frame["semantic"])
            for idx,box in enumerate(boxes):
                ref=build_reference(im,box)
                if ref is None: continue
                hit,why=current_support(ref,match_reference(ref,im),im,current)
                records.append(dict(step=step,slot=idx,source=verify_pixels(item,masks,box,[masks]),can_match=hit is not None,reason=why))
                print("ANCHOR",records[-1],flush=True)
    finally:
        detector.close(); env.close()
    (P/"jepa_reid_v24/sim/anchors.json").write_text(json.dumps(records,indent=2))
if __name__=="__main__" and "--anchors" in sys.argv:
    anchor_check()
