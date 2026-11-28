"""Paired Habitat branches at preselected archived states; decisions consume online RGB-D only."""
import copy
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MV = ROOT / "multiview_nav_20261128"
sys.path.insert(0, str(MV))
from run_multiview import ScanRecorder, environment, physical, FAMILY, read, digest
from multi_policy import located_cue
from multi_worker import start_worker
from map_views import visible_map_surfaces
from local_runtime import request
from routes import observation_routes, score_observation_routes
from gateway import ObservationGateway


def private_write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,allow_nan=False))
    tmp.replace(path)


def state(sim):
    s = sim.get_agent(0).state
    q = s.rotation
    return dict(xyz=s.position.tolist(),quat_xyzw=[float(k) for k in (q.x,q.y,q.z,q.w)])


def select(events):
    verify = [e for e in events if e.get("cue_selected") and e["anchors"] and not e["sites"]]
    result = [(e,"verify") for e in verify]
    for first,last in ((0,99),(100,199),(200,295)):
        candidates = [e for e in events if first<=e["step"]<=last and e.get("physical_scan")
            and e.get("gpt_mode")=="explore" and e.get("region_entered") and len(e["actions"])>=4
            and e.get("action_source")=="region" and e.get("commit_previous_id") is None]
        if candidates:
            result.append((candidates[0],"discover"))
    return {e["step"]:(e,m) for e,m in result}


def act_branch(sim, source_grid, seen, actions):
    grid = copy.deepcopy(source_grid)
    observed = set()
    trail = []
    collide = False
    old = sim.get_agent(0).state.position.copy()
    for act in actions:
        obs = sim.step({1:"move_forward",2:"turn_left",3:"turn_right"}[act])
        pose = physical.policy_pose(sim)
        grid.update(pose,obs["depth"])
        observed.update(visible_map_surfaces(grid,pose,1.))
        current = sim.get_agent(0).state.position.copy()
        collide = act==1 and (bool(obs.get("collided",False)) or np.linalg.norm(current-old)<.08)
        old = current
        trail.append(state(sim))
        if collide:
            break
    return dict(trail=trail,collision=bool(collide),surface_gain=len(observed-seen))


def snapshot(sim, scan, grid, seen, pose, e, mode, goal, channel, blocked):
    frames = list(scan.frames)
    views = [frames[j]["rgb"] for j in (1,0,3)]
    geometry, receipt = scan.packet(views,frames[0]["depth"],pose)
    assert receipt==e["physical_scan"], "source observations diverged"
    budget = 300-e["step"]
    anchors = []
    box = None
    if mode=="verify":
        box=dict(category=goal,view_index=e["cue_view_index"],bbox_norm=e["cue_bbox_norm"],candidate_id=e["cue_candidate_id"])
        anchors,_,_=located_cue(box,views,grid,pose,1.,geometry,90.)
        assert len(anchors)==1 and np.allclose(anchors[0]["anchor_xy"],e["cue_anchor_xy"],atol=1e-6)
    gate = ObservationGateway()
    admit,why = gate.admit(upper_mode=e["gpt_mode"],accepted_cue=box is not None,anchors=anchors,
        blocked=blocked,budget=budget,region=e["executed_region_id"],occupied=e.get("commit_previous_id") is not None)
    base = e["actions"][:min(budget,4)]
    assert base
    model="continue"
    geometry_id="continue"
    choices=[]
    gate_status=why
    proposal_status="not_requested"
    if admit:
        assert admit==mode
        p=observation_routes(grid,pose,1.,mode=mode,region_actions=e["actions"],budget=budget,
            observed_surfaces=seen,anchor_xy=anchors[0]["anchor_xy"] if anchors else None)
        proposal_status=p["status"]
        eligible,gate_status=gate.discoverable(p,pose,1.,budget) if mode=="discover" else (len(p["candidates"])>1,p["status"])
        if eligible:
            def infer(plans):
                if box:
                    return request(channel,mode="selected_box",view_index=box["view_index"],bbox_norm=box["bbox_norm"],plans=plans)
                return request(channel,mode="observe",views=views,goal=goal,plans=plans)["evidence"]
            # Source-frame context must belong to THIS decision before selected-box scoring.
            if box:
                request(channel,mode="observe",views=views,goal=goal,plans=[base])
            p=score_observation_routes(p,infer)
            model,geometry_id=p["selected_id"],p["geometry_id"]
        choices=p["candidates"]
    if not choices:
        choices=[dict(id="continue",actions=base,model_score=None,geometry_score=None,new_surfaces=None)]
    assert choices[0]["id"]=="continue" and list(choices[0]["actions"])==base
    source=sim.get_agent(0).state
    outcomes=[]
    try:
        for c in choices:
            sim.get_agent(0).set_state(source,reset_sensors=True)
            assert np.allclose(physical.policy_pose(sim),pose,atol=1e-6)
            result=act_branch(sim,grid,seen,c["actions"])
            outcomes.append(dict(id=c["id"],actions=c["actions"],geometry_score=c["geometry_score"],
                model_score=c["model_score"],map_forecast=c["new_surfaces"],**result))
    finally:
        sim.get_agent(0).set_state(source,reset_sensors=True)
    return dict(step=e["step"],mode=mode,box=box,approved=admit,gate=gate_status,proposal=proposal_status,
        selected=model,geometry=geometry_id,outcomes=outcomes)


def run_house(i,spec,channel):
    private=MV/"private_events"
    p,t=private/f"{i}.jsonl",private/f"{i}.poses.jsonl"
    accepted=read(MV/"runs"/f"{i}_multiview.json")
    assert digest(p)==accepted["private_events_sha256"] and digest(t)==accepted["private_trace_sha256"]
    events=[json.loads(s) for s in p.read_text().splitlines()]
    states=[json.loads(s) for s in t.read_text().splitlines()]
    frozen=select(events)
    ep,sim=environment(spec)
    scan=ScanRecorder(sim)
    records=[]
    try:
        start=physical.hs.AgentState()
        start.position=np.float32(ep["start_position"])
        start.rotation=physical.quat_from_coeffs(ep["start_rotation"])
        sim.get_agent(0).set_state(start)
        obs=scan.get_sensor_observations()
        pose=physical.policy_pose(sim)
        grid=physical.DepthGrid(pose[:2])
        grid.update(pose,obs["depth"])
        seen=set(visible_map_surfaces(grid,pose,1.))
        last_blocked=False
        for step,actual in enumerate(states[1:max(frozen)+1],1):
            last=pose
            obs=scan.step(actual["action"])
            pose=physical.policy_pose(sim)
            assert np.allclose(sim.get_agent(0).state.position,actual["xyz"],rtol=0,atol=1e-5)
            assert abs(math.remainder(pose[2]-actual["yaw"],math.tau))<1e-5
            grid.update(pose,obs["depth"])
            seen.update(visible_map_surfaces(grid,pose,1.))
            if actual["action"]=="move_forward":
                last_blocked=bool(obs.get("collided",False)) or math.dist(last[:2],pose[:2])<.08
                if last_blocked:
                    grid.record_contact(last)
            if step in frozen:
                e,mode=frozen[step]
                # Host resets the contact flag after each prior decision; the prior
                # log line is the authoritative blocked input even when a scan turns.
                decisions=[v for v in events if v.get("last_step",10**6)<=step-4 and "actions" in v]
                blocked=bool(decisions[-1]["contact"]) if decisions else False
                r=snapshot(sim,scan,grid,seen,pose,e,mode,ep["object_category"],channel,blocked)
                records.append(r)
                print(i,step,mode,r["gate"],len(r["outcomes"]),r["selected"],flush=True)
        return records
    finally:
        sim.close()


def main():
    os.umask(0o077)
    private=HERE/"probe_private"
    private.mkdir(exist_ok=True,mode=0o700)
    channel,worker=start_worker(ROOT,private/"frozen_worker.log")
    try:
        for i,spec in enumerate(read(FAMILY/"panel.json")):
            rows=run_house(i,spec,channel)
            private_write(private/f"branches_{i}.json",rows)
    finally:
        channel.close()
        worker.wait(timeout=60)


if __name__=="__main__":
    main()
