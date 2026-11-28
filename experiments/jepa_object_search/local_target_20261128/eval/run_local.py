"""Paired physical approach evaluation using the previous budget, sensors and scorer."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT/"approach_search_20261128/eval4"
sys.path[:0] = [str(HERE.parent), str(BASE)]
from physical_panel import (DepthGrid, PriorEvidenceMemory, ViewFollower, candidates, frontier_plans,
                            visible_map_surfaces, sensor_sim, policy_pose, measure, DATA, hs, quat_from_coeffs)
from local_runtime import pick, fast_continue, start_worker, plan_to_site
from local_search import LocalTargetTracker


def score_locations(sim, ep, records):
    """Offline-only distance diagnostics; the worker and the policy never see these."""
    from stopwatch.eval.geometry import _load_goal_sets, load_scene_instances, geodesic_distance
    ids = _load_goal_sets(DATA)["scoring_class_ids"]["hssdov"][ep["object_category"]]
    groups = [np.float32([v["agent_state"]["position"] for v in r["view_points"]])
              for r in load_scene_instances(DATA/"protocol/instance_viewpoints/hssd", "val", ep["scene_id"])
              if r["class_id"] in ids and r["view_points"]]
    result = {}
    for category, positions in records.items():
        closest = np.full(len(groups), np.inf)
        for pos in sorted(set(tuple(np.round(p, 4)) for p in positions)):
            p = np.float32(pos)
            if not sim.pathfinder.is_navigable(p, max_y_delta=.1):
                continue
            for i, group in enumerate(groups):
                # Only nearest-within-3m matters for the cutoffs below.
                near = group[np.linalg.norm(group-p, axis=1) < min(3.001, closest[i]+.001)]
                if len(near):
                    closest[i] = min(closest[i], geodesic_distance(sim, p, near))
        result[category] = {str(t):int(np.count_nonzero(closest<t)) for t in (.25,.5,1.,2.,3.)}
    return result


def episode_run(sim, ep, spec, channel, log):
    sim.seed(1128)
    state=hs.AgentState()
    state.position=np.float32(ep["start_position"])
    state.rotation=quat_from_coeffs(ep["start_rotation"])
    sim.get_agent(0).set_state(state)
    assert sim.pathfinder.is_navigable(state.position,max_y_delta=.08)
    obs, pose=sim.get_sensor_observations(), policy_pose(sim)
    grid, memory, tracker=DepthGrid(pose[:2]), PriorEvidenceMemory(), LocalTargetTracker()
    positions=[state.position.tolist()]
    coverage={k:[] for k in ("regional_motion","local_motion","local_arrivals","selected_local_sites",
                             "proposed_local_sites","proposed_region_sites","preferred_region_sites")}
    stats=Counter()
    counts={k:Counter() for k in ("releases","geometry","surfaces","local_physical_actions")}
    grid.update(pose,obs["depth"])
    memory.mark_view(pose[:2],pose[2],visible_map_surfaces(grid,pose,1.))
    def act(i, phase):
        nonlocal pose,obs
        obs=sim.step({1:"move_forward",2:"turn_left",3:"turn_right"}[i])
        pose=policy_pose(sim)
        grid.update(pose,obs["depth"])
        memory.mark_view(pose[:2],pose[2],visible_map_surfaces(grid,pose,1.))
        positions.append(sim.get_agent(0).state.position.tolist())
        stats["steps"]+=1
        stats[phase+"_steps"]+=1
        if phase in ("region", "local"):
            coverage["regional_motion" if phase=="region" else "local_motion"].append(positions[-1])
        if phase=="local":
            counts["local_physical_actions"][str(i)]+=1
        return obs["rgb"][:,:,:3].copy(), bool(obs.get("collided",False))
    def note(e):
        log.write(json.dumps(e, allow_nan=False)+"\n")
        log.flush()
    def audit_sites(regional, local, preferred):
        for name,points in (("proposed_region_sites",regional),("proposed_local_sites",local),
                            ("preferred_region_sites",[preferred["position"]] if preferred else [])):
            coverage[name].extend([x,positions[-1][1],-y] for x,y in points)
    def scan_decision(blocked):
        center=obs["rgb"][:,:,:3].copy()
        center_depth=obs["depth"].copy()
        right,_=act(3,"scan")
        act(2,"scan")
        left,_=act(2,"scan")
        act(3,"scan")
        remaining=budget-stats["steps"]
        basic=frontier_plans(grid,pose,1.,remaining)
        for turn in (0,-1,1,-2,2):
            rotation=[2 if turn>0 else 3]*abs(turn)
            for n in (8,3,1):
                basic.append((f"depth/{turn}/{n}",rotation+[1]*n))
        rows=candidates(grid,pose,1.,basic,remaining)
        return pick(tracker,memory,grid,pose,1.,rows,[right,center,left],ep["object_category"],
                    remaining,channel,center_depth,blocked=blocked,replay_memory=True,audit_sites=audit_sites)
    blocked=False
    budget=spec["budget"]
    while stats["steps"] < budget:
        dec=fast_continue(tracker,grid,pose,1.,budget-stats["steps"],blocked)
        fast=dec is not None
        region=None
        if fast:
            stats["rescan_saved"]+=1
        elif budget-stats["steps"] < 5:
            for _ in range(budget-stats["steps"]):
                act(2,"tail")
            break
        else:
            dec,region=scan_decision(blocked)
        blocked=False
        stats["decisions"]+=1
        stats[dec["event"]]+=1
        stats["inside_region"]+=int(dec["region_entered"])
        if not fast and (dec["region_entered"] or dec["action_source"]=="local"):
            stats["eligible_frames"]+=1
            for field in ("patches","anchors","sites"):
                stats["with_"+field]+=int(dec[field]>0)
        if dec.get("released"):
            rel=dec["released"]
            counts["releases"][rel["event"]]+=1
            if rel.get("geometry"):
                counts["geometry"][rel["geometry"]]+=1
            if rel["event"]=="arrived":
                coverage["local_arrivals"].append(positions[-1])
        if dec["event"]=="acquired":
            counts["surfaces"][str(dec["surface"])]+=1
            x,y=dec["site_xy"]
            coverage["selected_local_sites"].append([x,positions[-1][1],-y])
        if dec["action_source"]=="local":
            stats["max_age"]=max(stats["max_age"],dec["age"])
        if not dec["actions"]:
            note(dict(step=stats["steps"],**dec))
            act(2,"empty")
            continue
        follower=ViewFollower(pose,1.,dec["actions"])
        start=stats["steps"]
        for _ in range(min(len(dec["actions"])+8,budget-stats["steps"])):
            follower.x,follower.y,follower.yaw=pose
            a,arrived=follower.next_action()
            if a is None or arrived:
                break
            previous=pose
            old=sim.get_agent(0).state.position.copy()
            _,collision=act(a,dec["action_source"])
            if a==1 and (collision or np.linalg.norm(sim.get_agent(0).state.position-old)<.08):
                blocked=True
                stats["collisions"]+=1
                grid.record_contact(previous)
                if dec["action_source"]=="region" and region:
                    memory.mark_blocked(region["position"],region["heading"])
                break
        note(dict(step=start, last_step=stats["steps"], contact=blocked, **dec))
        if dec["action_source"]=="local":
            receipt=tracker.arrival(pose,lambda t:plan_to_site(grid,pose,1.,t))
            if receipt:
                coverage["local_arrivals"].append(positions[-1])
                counts["releases"]["arrived"]+=1
                note(dict(step=stats["steps"],**receipt))
    task=measure(sim,ep,positions)
    return dict(arm="local",scene=spec["scene"],episode=spec["id"],goal=ep["object_category"],
                steps=stats["steps"],stats=dict(stats),counts={k:dict(v) for k,v in counts.items()},
                distance_by_phase=score_locations(sim,ep,coverage),**task)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--indices",type=int,nargs="+",required=True)
    parser.add_argument("--panel-dir",type=Path,required=True)
    args=parser.parse_args()
    stage=args.panel_dir.resolve()
    for rel,digest in json.loads((stage/"local_lock.json").read_text()).items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:
            raise ValueError("changed frozen dependency: "+rel)
    panel=json.loads((stage/"panel.json").read_text())
    out=stage/"runs"
    private=stage/"private_events"
    out.mkdir(exist_ok=True)
    private.mkdir(mode=0o700,exist_ok=True)
    channel,process=start_worker(ROOT,private/f"model_{args.indices[0]}.log")
    try:
        for i in args.indices:
            spec=panel[i]
            dst=out/f"{i}_local.json"
            if dst.exists():
                continue
            shard=DATA/"eval/hssdov/content"/(spec["scene"]+".json.gz")
            assert hashlib.sha256(shard.read_bytes()).hexdigest()==spec["shard_sha256"]
            ep=next(e for e in json.load(gzip.open(shard))["episodes"] if str(e["episode_id"])==str(spec["id"]))
            sim=sensor_sim(ep)
            try:
                with (private/f"{i}.jsonl").open("a") as log:
                    row=episode_run(sim,ep,spec,channel,log)
                assert row["steps"]==300 and row["reachable_instances_at_start"]>=2
                tmp=dst.with_suffix(".tmp")
                tmp.write_text(json.dumps(row,indent=2)+"\n")
                tmp.replace(dst)
                print("finished",i,flush=True)
            finally:
                sim.close()
    finally:
        channel.close()
        process.wait(timeout=30)


if __name__=="__main__":
    main()
