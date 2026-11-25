"""Render labeled multi-view instance photographs from preassigned HSSD houses."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from PIL import Image
import habitat_sim as hs
from habitat_sim.utils.common import quat_from_angle_axis
import magnum as mn
import numpy as np


WORDS = set("""chair armchair bed sofa couch toilet television_receiver potted_plant lamp wall_art clock mirror picture_frame basket cabinet shelf stool table vase book desk refrigerator bathtub sink oven chandelier ceiling_lamp floor_lamp plant nightstand bookcase dresser bench""".split())
PRIORITY = ("clock", "wall_art", "picture_frame", "mirror", "chair", "lamp", "vase")
SIZE = 384


def open_sim(assets, house):
    cfg = hs.SimulatorConfiguration()
    cfg.scene_dataset_config_file = str(assets / "hssd-hab.scene_dataset_config.json")
    cfg.scene_id = house
    cfg.enable_physics = True
    ag = hs.agent.AgentConfiguration(height=1.4, radius=.18)
    for key, typ in (("rgb", hs.SensorType.COLOR), ("ids", hs.SensorType.SEMANTIC)):
        s = hs.CameraSensorSpec()
        s.uuid, s.sensor_type, s.resolution = key, typ, [SIZE, SIZE]
        s.position, s.hfov = mn.Vector3(0, 1.25, 0), mn.Deg(75)
        ag.sensor_specifications.append(s)
    sim = hs.Simulator(hs.Configuration(cfg, [ag]))
    nav = hs.NavMeshSettings()
    nav.set_defaults()
    nav.agent_height, nav.agent_radius, nav.include_static_objects = 1.4, .18, True
    if not sim.recompute_navmesh(sim.pathfinder, nav):
        raise RuntimeError(f"{house}: could not compute navigability")
    return sim


def inventory(sim, assets):
    with (assets / "metadata/fpmodels-with-decomposed.csv").open() as f:
        synsets = {r["id"]: r["main_wnsynsetkey"] or r["wnsynsetkey"] for r in csv.DictReader(f)}
    groups = defaultdict(list)
    objects = sim.get_rigid_object_manager().get_objects_by_handle_substring("")
    for sid, (name, obj) in enumerate(sorted(objects.items()), 1):
        obj.semantic_id = sid
        asset = next((k for k in re.findall(r"[a-fA-F0-9]{40}", name) if k in synsets), None)
        if asset is None:
            continue
        category = synsets[asset].split(".")[0]
        if category in WORDS:
            center = np.array(obj.transformation.transform_point(obj.aabb.center()), dtype=float)
            groups[category].append((sid, asset, center))
    return groups


def padded(box, ratio):
    x0, y0, x1, y1 = box
    dx, dy = (x1-x0)*ratio, (y1-y0)*ratio
    return (max(0, math.floor(x0-dx)), max(0, math.floor(y0-dy)),
            min(SIZE, math.ceil(x1+dx)), min(SIZE, math.ceil(y1+dy)))


def sample_object(sim, house, split, sid, asset, center, category, dest, rng):
    target = 6
    agent = sim.get_agent(0)
    rows, positions = [], []
    # Image goal is centered on a jittered look direction. Camera origin is always on the navmesh.
    az0 = rng.uniform(-math.pi, math.pi)
    for t in range(40):
        angle = az0 + t * math.pi * (3 - math.sqrt(5))
        r = rng.uniform(1.35, 3.1)
        nominal = center + [r*math.cos(angle), -0.75, r*math.sin(angle)]
        pos = np.asarray(sim.pathfinder.snap_point(nominal), dtype=float)
        if not np.isfinite(pos).all() or abs(pos[1]-center[1])>2.9:
            continue
        eye = pos + [0, 1.25, 0]
        d = center - eye
        flat = np.linalg.norm(d[[0,2]])
        if not .85 <= flat <= 3.8:
            continue
        az = math.degrees(math.atan2(-d[2], -d[0]))
        if any(abs((az-old+180)%360-180)<15 or np.linalg.norm(pos-p)<.35 for old,p in positions):
            continue
        yaw = math.atan2(-d[0], -d[2]) + rng.uniform(-.12, .12)
        pitch = math.atan2(d[1], flat) + rng.uniform(-.06,.06)
        # Keep the 1.25m camera height independent of the optical pitch.
        state = hs.AgentState()
        state.position = np.asarray(pos)
        state.rotation = quat_from_angle_axis(yaw, np.array([0.,1.,0.]))
        agent.set_state(state)
        state = agent.get_state()
        orient = state.rotation * quat_from_angle_axis(pitch, np.array([1.,0.,0.]))
        for sensor in state.sensor_states.values():
            sensor.rotation = orient
        agent.set_state(state, infer_sensor_states=False, reset_sensors=False)
        obs = sim.get_sensor_observations()
        yy, xx = np.where(obs["ids"] == sid)
        if len(xx)<170:
            continue
        box = tuple(map(int, (xx.min(), yy.min(), xx.max()+1, yy.max()+1)))
        x0,y0,x1,y1 = box
        if min(x1-x0, y1-y0)<12 or len(xx)/((x1-x0)*(y1-y0)) < .10:
            continue
        tight, context = padded(box,.15), padded(box,.80)
        stem = f"{sid:04d}_{len(rows):02d}"
        rel = Path("images") / split / house
        path = dest / rel
        path.mkdir(parents=True, exist_ok=True)
        frame = Image.fromarray(obs["rgb"][:,:,:3])
        for crop, name in ((tight, stem), (context, stem+"_ctx")):
            frame.crop(crop).save(path/(name+".jpg"), quality=93)
        heading = math.degrees(math.atan2(-math.cos(yaw), -math.sin(yaw)))
        rows.append(dict(split=split, house=house, instance=f"{house}:{sid}", category=category,
                         asset=asset, image=(rel/(stem+".jpg")).as_posix(),
                         context_image=(rel/(stem+"_ctx.jpg")).as_posix(), yaw_deg=az,
                         camera_heading_deg=heading, pitch_deg=math.degrees(pitch),
                         camera_xyz=eye.tolist(), pixels=len(xx), view=len(rows),
                         box_in_context=[x0-context[0], y0-context[1], x1-context[0], y1-context[1]],
                         bbox=list(box), tight_bbox=list(tight), context_bbox=list(context)))
        positions.append((az,pos.copy()))
        if len(rows)==target:
            break
    return rows


def collect(args):
    out = args.dest / "houses" / f"{args.split}_{args.house}.json"
    if out.exists():
        return
    seed = int(hashlib.sha256(("hssd-reid-v20|" + args.house).encode()).hexdigest()[:8],16)
    rng = np.random.default_rng(seed)
    sim = open_sim(args.assets, args.house)
    try:
        groups = inventory(sim, args.assets)
        categories = [w for w in PRIORITY if w in groups] + sorted(set(groups)-set(PRIORITY))
        for objs in groups.values():
            rng.shuffle(objs)
        rows = []
        # Sample multiple copies of every included category, including identical asset templates.
        for c in categories:
            if len(groups[c])<2:
                continue
            retained = []
            for sid, asset, center in groups[c][:8]:
                views = sample_object(sim,args.house,args.split,sid,asset,center,c,args.dest,rng)
                if len(views)>=2:
                    retained.extend(views)
                if len({r["instance"] for r in retained})>=5:
                    break
            if len({r["instance"] for r in retained})>=2:
                rows.extend(retained)
    finally:
        sim.close()
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows))
    tmp.replace(out)
    print(args.house,len(rows),len({r["instance"] for r in rows}),dict(Counter(r["category"] for r in rows)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("assets",type=Path)
    p.add_argument("dest",type=Path)
    p.add_argument("split")
    p.add_argument("house")
    collect(p.parse_args())
