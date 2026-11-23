"""Make a branch-independent HSSD goal catalogue using the navigable floor and rendered instances."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import habitat_sim as hs
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_coeffs
import magnum as mn
import numpy as np


# Annotated HSSD asset classes. Attributes such as "white" are not inferred from class names.
CATEGORIES = {
    "wall_art.n.01": "wall art", "picture_frame.n.01": "picture frame",
    "door.n.01": "door", "mirror.n.01": "mirror",
    "wall_clock.n.01": "clock", "alarm_clock.n.01": "clock", "mantel_clock.n.01": "clock", "clock.n.01": "clock",
    "ceiling_lamp.n.01": "lamp", "floor_lamp.n.01": "lamp", "table_lamp.n.01": "lamp", "wall_lamp.n.01": "lamp",
    "cabinet.n.01": "cabinet", "shelf.n.01": "shelf", "basket.n.01": "basket", "stool.n.01": "stool",
}
RESOLUTION, EYE_HEIGHT, GRID = 128, 1.25, 0.4


def load_scene(raw, house, gpu):
    cfg = hs.SimulatorConfiguration()
    cfg.gpu_device_id = gpu
    cfg.scene_dataset_config_file = str(raw / "hssd-hab.scene_dataset_config.json")
    cfg.scene_id, cfg.enable_physics, cfg.allow_sliding = house, True, False
    c = hs.agent.AgentConfiguration(height=1.4, radius=.18)
    for name, kind in (("seg", hs.SensorType.SEMANTIC), ("depth", hs.SensorType.DEPTH)):
        spec = hs.CameraSensorSpec()
        spec.uuid, spec.sensor_type, spec.resolution = name, kind, [RESOLUTION, RESOLUTION]
        spec.position, spec.hfov = mn.Vector3(0, EYE_HEIGHT, 0), mn.Deg(90)
        c.sensor_specifications.append(spec)
    sim = hs.Simulator(hs.Configuration(cfg, [c]))
    nav = hs.NavMeshSettings()
    nav.set_defaults()
    nav.agent_height, nav.agent_radius, nav.include_static_objects = 1.4, .18, True
    if not sim.recompute_navmesh(sim.pathfinder, nav):
        raise RuntimeError("House navigation mesh unavailable")
    return sim


def independent_positions(pf, center, bottom):
    # Fixed world grid, independent of all robot routes, RGB histories and branch returns.
    x0, z0 = np.round(center[[0, 2]] / GRID) * GRID
    seen = set()
    for ix in range(-5, 6):
        for iz in range(-5, 6):
            x, z = x0 + ix * GRID, z0 + iz * GRID
            if not .45 <= np.hypot(x - center[0], z - center[2]) <= 2.0:
                continue
            # Wall and ceiling fixtures can sit well above the local navigable floor.
            for anchor_y in (bottom, center[1] - EYE_HEIGHT, center[1] - 2.4):
                point = np.asarray(pf.snap_point(np.array([x, anchor_y, z], dtype=np.float32)))
                if not np.isfinite(point).all() or np.hypot(point[0] - x, point[2] - z) > .10:
                    continue
                if not -.35 <= center[1] - point[1] <= 3.5 or not pf.is_navigable(point):
                    continue
                key = tuple(np.round(point, 3))
                if key not in seen:
                    seen.add(key)
                    yield point


def derive(raw, house, out, gpu):
    with (raw / "semantics/objects.csv").open() as f:
        assets = {r["id"]: r for r in csv.DictReader(f)}
    sim = load_scene(raw, house, gpu)
    records, counts = [], Counter()
    try:
        objects = sorted(sim.get_rigid_object_manager().get_objects_by_handle_substring("").items())
        for sid, (_, obj) in enumerate(objects, 1):
            obj.semantic_id = sid
        for sid, (name, obj) in enumerate(objects, 1):
            asset = assets.get(name.split(":")[0].rstrip("_"), {})
            synset = (asset.get("main_wnsynsetkey") or asset.get("wnsynsetkey") or "").split(",")[0].strip()
            if synset not in CATEGORIES:
                continue
            box = hs.geo.get_transformed_bb(obj.root_scene_node.cumulative_bb, obj.transformation)
            center, bottom = np.asarray(box.center(), dtype=float), float(box.min[1])
            views = []
            for point in independent_positions(sim.pathfinder, center, bottom):
                delta = center - point - np.array([0., EYE_HEIGHT, 0.])
                yaw = math.atan2(-delta[0], -delta[2])
                pitch = np.clip(math.atan2(delta[1], np.hypot(delta[0], delta[2])), -math.pi/6, math.pi/6)
                yrot = quat_from_angle_axis(yaw, np.array([0., 1., 0.]))
                xrot = quat_from_angle_axis(float(pitch), np.array([1., 0., 0.]))
                state = hs.AgentState()
                state.position, state.rotation = point, yrot
                agent = sim.get_agent(0)
                agent.set_state(state)
                state = agent.get_state()
                for sensor_pose in state.sensor_states.values():
                    sensor_pose.rotation = yrot * xrot
                agent.set_state(state, infer_sensor_states=False)
                obs = sim.get_sensor_observations()
                mask = obs["seg"] == sid
                area = int(mask.sum())
                near = int(np.sum(mask & (obs["depth"] > 0) & (obs["depth"] <= 2.0)))
                counts["rendered_views"] += 1
                if area < 48 or near < 32:
                    continue
                views.append({"agent_state": {"position": point.tolist(), "rotation": quat_to_coeffs(yrot).tolist()},
                              "sensor_rotation": quat_to_coeffs(yrot * xrot).tolist(),
                              "visible_pixels": area, "near_pixels": near})
            views.sort(key=lambda v: -v["visible_pixels"])
            records.append(dict(object_name=name, object_id=int(obj.object_id), object_category=CATEGORIES[synset],
                                asset_synset=synset, position=center.tolist(), view_points=views))
            counts["instances"] += 1
            counts["instances_with_views"] += bool(views)
            counts["valid_views"] += len(views)
    finally:
        sim.close()
    result = {"house": house, "source": "derived_hssd_render_grid_v1", "goals": records,
              "counts": counts, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(result, f, separators=(",", ":"), allow_nan=False)
    os.replace(tmp, out)
    print(house, dict(counts), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--house", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    dest = a.output / (a.house + ".json.gz")
    if not dest.exists():
        derive(a.raw, a.house, dest, a.gpu)
