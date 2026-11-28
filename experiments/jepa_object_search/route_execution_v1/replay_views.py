"""Paired, network-disabled HM3D test at every archived V18 selected observation view."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import quaternion

from agents.navigator import PathFollower
from evaluation.main.evaluator import _env_action
from verify_habitat import replay_until, scene
from verify_rank import deny_ip_sockets
from point_executor import ViewFollower, route_geometry


def read(path):
    return json.loads(path.read_bytes())


def measure(sim, origin, rotation, goal, yaw):
    st = sim.get_agent_state()
    local = rotation.T @ (st.position - origin)
    direction = rotation.T @ quaternion.as_rotation_matrix(st.rotation) @ [0., 0., -1.]
    distance = math.hypot(-local[2] - goal[0], -local[0] - goal[1])
    angle = abs(math.degrees(math.remainder(math.atan2(-direction[0], -direction[2]) - yaw, math.tau)))
    return float(distance), float(angle)


def reference_geometry(sim, actions):
    """Scorer-only diagnostic. The policy never receives mesh distances or these locations."""
    import habitat_sim
    state = sim.get_agent_state()
    path, _ = route_geometry((0., 0., 0.), 1., actions)
    x, y = path[-1]
    goal = state.position + quaternion.as_rotation_matrix(state.rotation) @ [-y, 0., -x]
    pf = sim.pathfinder
    island = pf.get_island(state.position)
    nearest = np.asarray(pf.snap_point(goal, island), dtype=float)
    query = habitat_sim.ShortestPath()
    query.requested_start, query.requested_end = state.position, nearest
    ok = np.isfinite(nearest).all() and pf.find_path(query)
    horizontal_displacement = float(np.linalg.norm((nearest - goal)[[0, 2]])) if ok else None
    return dict(horizontal_gap_to_snapped_mesh_m=horizontal_displacement,
                vertical_gap_to_snapped_mesh_m=float(abs(nearest[1]-goal[1])) if ok else None,
                geodesic_to_snapped_goal_m=float(query.geodesic_distance) if ok else None)


def run_arm(env, frame, actions, available, arm, native_reach):
    pose = (0., 0., 0.)
    path, yaw = route_geometry(pose, 1., actions)
    st = env.sim.get_agent_state()
    origin, rotation = st.position.copy(), quaternion.as_rotation_matrix(st.rotation)
    follower = ViewFollower(pose, 1., actions) if arm == "selected_view" else PathFollower(1., native_reach)
    follower.set_path(path)
    limit = min(available, len(actions) + 8)
    issued = collisions = consecutive = 0
    reason = "local_budget"
    rgb = frame["rgb"][::16, ::16, :3].astype(np.int16)
    while issued < limit:
        action, arrived = follower.next_action()
        if action is None or arrived:
            reason = "estimated_arrival" if arrived else "lost_path"
            break
        frame = env.step(_env_action(action))
        follower.dead_reckon(action)
        issued += 1
        collisions += int(action == 1 and env.sim.previous_step_collided)
        current = frame["rgb"][::16, ::16, :3].astype(np.int16)
        delta = np.mean(np.abs(current - rgb))
        consecutive = consecutive + 1 if action == 1 and delta < float(os.getenv("MAPPING_STUCK_RGB_DELTA", "1.0")) else 0
        rgb = current
        if consecutive >= int(os.getenv("MAPPING_STUCK_CONFIRM_STEPS", "2")):
            reason = "native_rgb_stall_yield"
            break
    distance, angle = measure(env.sim, origin, rotation, path[-1], yaw)
    return dict(actions=issued, reason=reason, collision_actions=collisions, position_error_m=round(distance, 4),
                yaw_error_deg=round(angle, 3), position_ok=distance <= .125 + 1e-5,
                view_ok=distance <= .125 + 1e-5 and angle <= 15.001)


def main(project, dest):
    deny_ip_sockets()
    pilot = project / "mapping_runtime/pilots/jepa_confirm_v18_first2"
    results = []
    sources = []
    native_reach = float(os.getenv("NAV_REACH_M", ".8"))
    for row in read(pilot / "panel.json")["rows"]:
        ref = read(pilot / "comparison/accepted/second" / (row["episode_id"] + ".json"))
        source = Path(ref["result"])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == ref["result_sha256"]
        sources.append(ref["result_sha256"])
        doc = read(source)
        ep, = doc["episodes"]
        all_steps = ep["paper"]["steps"]
        records = [json.loads(x) for x in (source.parent / "agent/v13_routes.jsonl").read_bytes().splitlines()]
        env, item, data = scene(doc)
        try:
            for event in records:
                if event["event"] != "view_track_start":
                    continue
                step = event["step"]
                outputs = {}
                for arm in ("official_follower", "selected_view"):
                    frame = replay_until(env, item, data, all_steps, step)
                    geometry = reference_geometry(env.sim, event["prefix"])
                    outputs[arm] = run_arm(env, frame, event["prefix"], 300-step, arm, native_reach)
                results.append(dict(episode_ordinal=row["ordinal"], source_step=step,
                                    prefix_length=len(event["prefix"]), audit_geometry=geometry, arms=outputs))
        finally:
            env.close()
    assert len(results) == 9
    totals = {arm: {k: sum(v["arms"][arm][k] for v in results) for k in
                      ("position_ok", "view_ok", "actions", "collision_actions")}
              for arm in ("official_follower", "selected_view")}
    output = dict(source_sha256=sources, selected_views=len(results), native_reach_m=native_reach,
                  same_pose_view_budget=True, target_labels_used_by_controller=False,
                  protocol="Habitat full scene; paired archived poses and selected prefix path, equal local budget. Native RGB stall terminates both arms. Odometry during short branch; no live SLAM, upper policy, or paid API. Physical simulator pose only for resetting and scoring.",
                  totals=totals, results=results)
    dest.write_text(json.dumps(output, indent=2) + "\n")
    print(totals)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
