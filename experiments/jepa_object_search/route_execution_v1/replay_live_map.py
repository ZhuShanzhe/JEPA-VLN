"""Causal counterfactual on archived failed, geometrically reachable observation selections."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import quaternion

from agents import navigator as nav
from evaluation.main.evaluator import _env_action, _reset_environment
from mapping.client import MappingClient
from point_executor import ViewFollower, route_geometry
from replay_views import measure
from verify_habitat import scene


def snapshot(client, source, current, actions, ruler):
    frames = client.get_frame_points(stride=6)
    if not frames:
        return None
    all_poses, ids = client.get_all_poses()
    poses = dict(zip(ids, all_poses)) if all_poses is not None else {}
    if source not in poses or current not in poses:
        return None
    rotation = nav.gravity_alignment(np.stack(list(poses.values())), cam_up=nav.mount_compensated_cam_up())
    grid = nav.OccupancyGrid.from_frame_points(frames, rotation, unit_per_m=ruler)
    if grid is None:
        return None
    scale = (1./ruler if ruler else 1./grid.unit_per_m)
    local = lambda fid: nav.pose_to_yaw_2d(poses[fid], rotation)
    pose = local(current)
    waypoints, yaw = route_geometry(local(source), scale, actions)
    return grid, pose, scale, waypoints, yaw


def capture(client, frame):
    return int(client.capture_keyframe(np.ascontiguousarray(frame["rgb"][...,:3]))["frame_id"])


def flush(client):
    client.flush_map()
    if not client.wait_idle(timeout=120):
        raise TimeoutError("geometry mapper busy after flush")


def run(client, env, item, data, timeline, row):
    start = row["step"]
    actions = row["prefix"]
    budget = min(len(actions) + 8, 300 - start)
    client.reset_map()
    frame = _reset_environment(env, item, data)
    calibration = {}
    for i, step in enumerate(timeline[:start]):
        if (start-i) % 3 == 0:
            fid=capture(client, frame)
            # Original MostBench GPS is used only for distances determining map scale.
            # It never supplies an execution pose, obstacle location or goal coordinate.
            calibration[fid] = np.asarray(step["position_before"], dtype=float)
            if i % 6 == 0:
                client.wait_idle(timeout=120)
        if step["action_role"] == "navigate":
            frame = env.step(_env_action(step["action"]))
    state = env.sim.get_agent_state()
    origin, R = state.position.copy(), quaternion.as_rotation_matrix(state.rotation)
    desired, desired_yaw = route_geometry((0.,0.,0.),1.,actions)
    source = capture(client,frame)
    flush(client)
    mats, ids = client.get_all_poses()
    anchor_poses = dict(zip(ids,mats)) if mats is not None else {}
    observed = sorted(set(calibration) & set(anchor_poses))
    pairs=[]
    for a,b in zip(observed,observed[1:]):
        distance=np.linalg.norm(calibration[a]-calibration[b])
        delta=np.linalg.norm(anchor_poses[a][:3,3]-anchor_poses[b][:3,3])
        if .2 < distance < 1.1 and delta > .001:
            pairs.append(delta/distance)
    initial_ruler=float(np.median(pairs)) if len(pairs)>=3 else None
    geom = snapshot(client, source, source, actions, initial_ruler)
    issued = collisions = repairs = fixes = 0
    reason = "initial_map_unavailable"
    if geom is not None:
        grid, pose, scale, path, yaw = geom
        follower = ViewFollower(pose, scale, actions, allow_staging=True)
        ruler = 1./scale
        prev_rgb = frame["rgb"][::16,::16,:3].astype(np.int16)
        stationary = 0
        pending_repair = False
        blocks = []
        last = source
        needs_fix = False
        reason = "local_budget"
        while issued < budget:
            action, arrived = follower.next_action()
            if needs_fix or arrived or pending_repair:
                flush(client)
                updated = snapshot(client,source,last,actions,ruler)
                if updated is None:
                    reason = "map_location_unavailable"
                    break
                grid, pose, scale, path, yaw = updated
                fixes += 1
                follower.goal_xy, follower.final_yaw = path[-1], yaw
                correction = math.dist((follower.x, follower.y),pose[:2])*scale
                follower.x, follower.y, follower.yaw = pose
                action, arrived = follower.next_action()
                if math.dist(follower.goal_xy, pose[:2])*scale <= .125:
                    angle = math.remainder(yaw - pose[2],math.tau)
                    if abs(angle) <= math.pi/12+1.e-8:
                        reason = "current_map_pose_view_reached"
                        break
                    follower.set_path([follower.goal_xy])
                elif pending_repair or correction > .25 or arrived:
                    if pending_repair:
                        blocks.append(tuple(np.asarray(pose[:2])+.25/scale*np.array([math.cos(pose[2]), math.sin(pose[2])])))
                    if not follower.detour(grid,pose,blocks):
                        reason = follower.repair_stop or "no_observed_detour"
                        break
                    repairs += 1
                needs_fix = pending_repair = False
                action, arrived = follower.next_action()
            if action is None:
                reason = "native_path_ended"
                break
            frame = env.step(_env_action(action))
            follower.dead_reckon(action)
            issued += 1
            collisions += int(action == 1 and env.sim.previous_step_collided) # scorer only
            last = capture(client,frame)
            rgb = frame["rgb"][::16,::16,:3].astype(np.int16)
            stationary = stationary+1 if action==1 and np.mean(abs(rgb-prev_rgb))<1. else 0
            prev_rgb = rgb
            pending_repair = stationary >= 2
            needs_fix = issued % 5 == 0
    err, angle = measure(env.sim, origin,R, desired[-1],desired_yaw)
    return dict(step=start,actions=issued, calibration_segments=len(pairs),physical_position_error_m=err, physical_yaw_error_deg=angle,
                view_ok=err<=.125+1.e-5 and angle<=15.001, reason=reason,
                scored_collision_actions=collisions, local_map_fixes=fixes, local_map_repairs=repairs)


def main(project, outfile, mesh):
    client = MappingClient(port=18674,timeout=180)
    geometry = json.loads(mesh.read_bytes())["results"]
    results=[]
    pilot=project/"mapping_runtime/pilots/jepa_confirm_v18_first2"
    for r in json.loads((pilot/"panel.json").read_bytes())["rows"]:
        ref=json.loads((pilot/"comparison/accepted/second"/(r["episode_id"]+".json")).read_bytes())
        result=Path(ref["result"])
        assert hashlib.sha256(result.read_bytes()).hexdigest()==ref["result_sha256"]
        doc=json.loads(result.read_bytes())
        ep,=doc["episodes"]
        timeline=ep["paper"]["steps"]
        events=[json.loads(x) for x in (result.parent/"agent/v13_routes.jsonl").read_bytes().splitlines()]
        env,item,data=scene(doc)
        try:
            for e in events:
                if e["event"]!="view_track_start": continue
                g,=[v for v in geometry if v["source_step"]==e["step"] and v["episode_ordinal"]==r["ordinal"]]
                if g["arms"]["selected_view"]["view_ok"] or g["audit_geometry"]["horizontal_gap_to_snapped_mesh_m"]>.125:
                    continue
                answer=run(client,env,item,data,timeline,e)
                results.append(dict(ordinal=r["ordinal"],**answer))
                outfile.write_text(json.dumps(results,indent=2)+"\n")
                print(results[-1],flush=True)
        finally:
            env.close()
    client.close()
    assert len(results)==3


if __name__=="__main__":
    main(*map(Path,sys.argv[1:]))
