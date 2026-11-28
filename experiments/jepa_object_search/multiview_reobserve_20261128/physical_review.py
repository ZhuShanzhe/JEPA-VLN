"""Replay every fixed rejected site from online maps; score instance labels after rollout."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
MV=ROOT/"multiview_nav_20261128"
sys.path.insert(0,str(MV))
from run_multiview import ScanRecorder, environment, physical, FAMILY, read, digest
from multi_policy import located_cue
from cue_service import CueSelector
from local_runtime import propose_sites
from reobserve import Reobserve

ACT={1:"move_forward",2:"turn_left",3:"turn_right"}
PRIVATE=HERE/"private"


def write(p,x):
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")


def state_dict(state):
    q=state.rotation
    return dict(xyz=state.position.tolist(),quat_xyzw=[float(v) for v in (q.x,q.y,q.z,q.w)])


def detect_frame(cue,frame,goal):
    with tempfile.TemporaryDirectory(dir=PRIVATE) as d:
        p=Path(d)/"observed.png"
        Image.fromarray(frame).save(p)
        # The frozen worker takes triplets. Identical input copies count as ONE observation.
        rows=cue._propose([p,p,p],goal)[0]
    limits=np.array([frame.shape[1],frame.shape[0]]*2)
    return [dict(category=goal,bbox_norm=(b["xyxy"]/limits).tolist()) for b in rows]


def branch(sim, grid, pose, anchor, goal, remaining, cue):
    control=Reobserve(anchor,goal)
    history=[]
    while not control.end:
        a=control.next_action(grid,pose,1.,remaining-len(history))
        if a is None: break
        obs=sim.step(ACT[a])
        pose=physical.policy_pose(sim)
        grid.update(pose,obs["depth"])
        boxes=detect_frame(cue,obs["rgb"][:,:,:3],goal)
        match=control.observe(grid,pose,1.,obs["depth"],boxes,obs.get("collided",False))
        rec=dict(action=ACT[a],state=state_dict(sim.get_agent(0).state),candidate=match,
                 boxes=len(boxes),collision=bool(obs.get("collided",False)))
        if match is not None:
            frame=obs["rgb"][:,:,:3]
            fresh=dict(match,view_index=1,candidate_id="fresh")
            # Only the central pose/depth are used by selected view 1.
            new,_,_=located_cue(fresh,[frame]*3,grid,pose,1.,[(pose,obs["depth"])]*3,90.)
            sites=propose_sites(grid,pose,1.,new,min(24,remaining-len(history)-1))
            rec["strict_sites"]=len(sites)
            if sites:
                control.finish("strict_site_available")
        history.append(rec)
    return dict(actions=history,termination=control.end)


def observe_group(arm,i,spec,cue):
    path=MV if arm=="online" else ROOT/"committed_inspect_20261128"
    fn="multiview" if arm=="online" else "committed"
    paths=path/"private_events"
    events,trace=paths/f"{i}.jsonl",paths/f"{i}.poses.jsonl"
    source=read(path/"runs"/f"{i}_{fn}.json")
    assert digest(events)==source["private_events_sha256"] and digest(trace)==source["private_trace_sha256"]
    logs=[json.loads(x) for x in events.read_text().splitlines()]
    states=[json.loads(x) for x in trace.read_text().splitlines()]
    grants={e["step"]:e for e in logs if (e.get("cue_selected") and e["anchors"] and not e["sites"] if arm=="online" else e["event"]=="acquired")}
    if not grants: return []
    ep,sim=environment(spec)
    goal=ep["object_category"]
    scan=ScanRecorder(sim)
    results=[]
    try:
        start=physical.hs.AgentState()
        start.position=np.float32(ep["start_position"])
        start.rotation=physical.quat_from_coeffs(ep["start_rotation"])
        sim.get_agent(0).set_state(start)
        obs=scan.get_sensor_observations()
        pose=physical.policy_pose(sim)
        grid=physical.DepthGrid(pose[:2])
        grid.update(pose,obs["depth"])
        for step,t in enumerate(states[1:max(grants)+1],1):
            previous=pose
            obs=scan.step(t["action"])
            pose=physical.policy_pose(sim)
            assert np.allclose(sim.get_agent(0).state.position,t["xyz"],rtol=0,atol=1e-5)
            assert abs(math.remainder(pose[2]-t["yaw"],math.tau))<1e-5
            grid.update(pose,obs["depth"])
            if t["action"]=="move_forward" and (obs.get("collided",False) or math.dist(previous[:2],pose[:2])<.08):
                grid.record_contact(previous)
            if step not in grants: continue
            e=grants[step]
            frames=list(scan.frames)
            views=[frames[j]["rgb"] for j in (1,0,3)]
            geometry,pairs=scan.packet(views,frames[0]["depth"],pose)
            if arm=="online":
                assert pairs==e["physical_scan"]
                choice=dict(category=goal,view_index=e["cue_view_index"],bbox_norm=e["cue_bbox_norm"],candidate_id=e["cue_candidate_id"])
            else:
                choice=cue.select(views,goal)
                assert cue.last_receipt["event"]=="prior_cache"
            row=dict(arm=arm,house=i,step=step,goal=goal,choice=choice)
            if choice is None:
                row["status"]="no_accepted_box"
            else:
                anchors,_,_=located_cue(choice,views,grid,pose,1.,geometry,90.)
                sites=propose_sites(grid,pose,1.,anchors,min(24,300-step))
                if arm=="online":
                    assert len(anchors)==e["anchors"] and len(sites)==e["sites"]
                    assert np.allclose(anchors[0]["anchor_xy"],e["cue_anchor_xy"],atol=1e-6)
                if not anchors:
                    row["status"]="no_measured_anchor"
                elif sites:
                    row["status"]="already_has_site"
                else:
                    original=sim.get_agent(0).state
                    row["status"]="reobserve"
                    row["source_state"]=states[step-(3,4,1)[choice["view_index"]]]
                    row["start_state"]=states[step]
                    try:
                        row.update(branch(sim,copy.deepcopy(grid),pose,anchors[0],goal,300-step,cue))
                        row["equal_budget_continuation"]=states[step+len(row["actions"])]
                    finally:
                        sim.get_agent(0).set_state(original,reset_sensors=True)
            results.append(row)
            print(arm,i,step,row["status"],row.get("termination"),len(row.get("actions",[])),flush=True)
    finally:
        sim.close()
    return results


def run():
    cue=CueSelector(PRIVATE/"cue",allow_new=False)
    try:
        for arm in ("online","historical"):
            cases=[r for i,spec in enumerate(read(FAMILY/"panel.json")) for r in observe_group(arm,i,spec,cue)]
            assert len(cases)==(8 if arm=="online" else 4)
            write(PRIVATE/(arm+".json"),cases)
    finally:
        cue.close()
    assert not (PRIVATE/"cue/cue_calls.jsonl").exists()


def score():
    from review_interface import box_match
    from visibility_review import configured_sim, DATA, _load_goal_sets, load_scene_instances, instance_lut
    accepted=read(MV/"interface_review.json")
    rows=[]
    for arm in ("online","historical"):
        cases=read(PRIVATE/(arm+".json"))
        for i,spec in enumerate(read(FAMILY/"panel.json")):
            here=[r for r in cases if r["house"]==i]
            if not here: continue
            ep,plain=environment(spec)
            plain.close()
            classes=_load_goal_sets(DATA)["scoring_class_ids"]["hssdov"][ep["object_category"]]
            records=[r for r in load_scene_instances(DATA/"protocol/instance_viewpoints/hssd","val",ep["scene_id"])
                     if r["class_id"] in classes and r["view_points"]]
            targets=[r["semantic_id"] for r in records]
            sim=configured_sim(ep)
            lut=instance_lut(sim,"hssd",DATA/"vocab")
            manager=sim.get_rigid_object_manager()
            objects={o.semantic_id:o for o in (manager.get_object_by_handle(k) for k in manager.get_object_handles())}
            for r in records:
                sid=r["semantic_id"]
                assert lut[sid]==r["class_id"]+1
                assert Path(objects[sid].creation_attributes.handle).name.removesuffix(".object_config.json")==r["template_name"]
            expected=[s for s in accepted["houses"][i]["selected_cues"] if s["anchored"] and not s["site_offered"]]
            if arm=="online":
                assert len(here)==len(expected)
            def mask(t):
                cam=physical.hs.AgentState()
                cam.position=np.array(t["xyz"],dtype=np.float32)
                cam.rotation=physical.quat_from_coeffs(np.array(t["quat_xyzw"]))
                sim.get_agent(0).set_state(cam,reset_sensors=True)
                raw=np.asarray(sim.get_sensor_observations()["instance_mask"])
                return np.where(np.isin(raw,targets),raw,0)
            try:
                for n,case in enumerate(here):
                    public=dict(arm=arm,goal=case["goal"],status=case["status"])
                    if case["status"]=="reobserve":
                        initial=mask(case["source_state"])
                        sid,iou,_=box_match(initial,case["choice"]["bbox_norm"])
                        valid=bool(sid and iou>=.25)
                        if arm=="online":
                            assert valid==expected[n]["valid_box"] and round(iou,4)==expected[n]["iou_same_instance"]
                            assert expected[n]["view"]==("right","center","left")[case["choice"]["view_index"]]
                        actions=case["actions"]
                        baseline=mask(case["equal_budget_continuation"])
                        follow=mask(actions[-1]["state"] if actions else case["start_state"])
                        confirmations=[]
                        same_instance_pixels=[]
                        for action in actions:
                            action_mask=mask(action["state"])
                            if valid:
                                same_instance_pixels.append(int(np.count_nonzero(action_mask==sid)))
                            if action["candidate"]:
                                sid2,alignment,_=box_match(action_mask,action["candidate"]["bbox_norm"])
                                confirmations.append(bool(valid and sid2==sid and alignment>=.25))
                        public.update(original_valid=valid,source_view=("right","center","left")[case["choice"]["view_index"]],
                            actions=len(actions),forward=sum(a["action"]=="move_forward" for a in actions),
                            collisions=sum(a["collision"] for a in actions),matched_fresh=len(confirmations),
                            same_instance_matches=sum(confirmations),termination=case["termination"],
                            initial_pixels=int(np.count_nonzero(initial==sid)) if valid else None,
                            final_pixels=int(np.count_nonzero(follow==sid)) if valid else None,
                            best_after_move_pixels=max(same_instance_pixels,default=0) if valid else None,
                            baseline_equal_actions_pixels=int(np.count_nonzero(baseline==sid)) if valid else None)
                    rows.append(public)
            finally:
                sim.close()
    report=dict(code=digest(HERE/"reobserve.py"),runner=digest(Path(__file__)),new_paid_calls=0,
                endpoint="paired short physical branches; same-instance boxes require center and visible-envelope IoU >= 0.25",
                rows=rows)
    write(HERE/"physical_review.json",report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    os.umask(0o077)
    PRIVATE.mkdir(exist_ok=True,mode=0o700)
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["run","score"])
    {"run":run,"score":score}[p.parse_args().action]()
