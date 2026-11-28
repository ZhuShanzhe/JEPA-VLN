"""Render evaluator-only instance masks at the actual recorded camera poses after each rollout."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import sys

import habitat_sim as hs
import magnum as mn
from habitat_sim.utils.common import quat_from_coeffs
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "a_hssdov_external"
sys.path[:0] = [str(ROOT/"src/stopwatch"), str(ROOT/"local_target_20261128/localvision")]
from local_vision import BOXES
from stopwatch.data.semantics import instance_lut
from stopwatch.eval.geometry import _load_goal_sets, load_scene_instances, geodesic_distance

PIXELS = 192 * 192
THRESHOLDS = (.002, .01, .03)


def lines(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def configured_sim(ep, check_rgb=False):
    basedir = DATA/"runtime_check/scenes"
    cfg = hs.SimulatorConfiguration()
    cfg.scene_dataset_config_file = str(basedir/ep["scene_dataset_config"])
    cfg.scene_id = str(basedir/ep["scene_id"])
    cfg.enable_physics = True
    cfg.load_semantic_mesh = True
    cfg.use_semantic_textures = False
    sensor = hs.CameraSensorSpec()
    sensor.uuid = "instance_mask"
    sensor.sensor_type = hs.SensorType.SEMANTIC
    sensor.hfov = 90
    sensor.resolution = [192, 192]
    sensor.position = mn.Vector3(0, 1.31, 0)
    agent = hs.agent.AgentConfiguration(height=1.41, radius=.17)
    agent.sensor_specifications = [sensor]
    if check_rgb:
        color = hs.CameraSensorSpec()
        color.uuid = "rgb"
        color.sensor_type = hs.SensorType.COLOR
        color.hfov = sensor.hfov
        color.resolution = sensor.resolution
        color.position = sensor.position
        agent.sensor_specifications = [sensor, color]
    return hs.Simulator(hs.Configuration(cfg, [agent]))


def reference_proximity(ep, trajectory, rows):
    # Geodesic checks run with the original, physically configured trial simulator.
    sys.path.insert(0,str(ROOT/"approach_search_20261128/eval4"))
    from physical_panel import sensor_sim
    sim = sensor_sim(ep)
    views = {r["semantic_id"]:np.float32([v["agent_state"]["position"] for v in r["view_points"]]) for r in rows}
    cached, result = {}, []
    try:
        for state in trajectory:
            key = tuple(state["xyz"])
            if key not in cached:
                p = np.float32(key)
                close = set()
                for sid, points in views.items():
                    shortlist = points[np.linalg.norm(points-p,axis=1)<.25001]
                    if len(shortlist) and geodesic_distance(sim,p,shortlist)<.25:
                        close.add(sid)
                cached[key] = close
            result.append(cached[key])
    finally:
        sim.close()
    return result


def summary_for_arm(sim, targets, trajectory, event_path, proximity):
    events = lines(event_path)
    starts = {e["step"]: e for e in events if e.get("event") == "acquired"}
    resumed = {}
    for e in events:
        if e.get("action_source") == "local":
            resumed[e["goal_id"]] = e.get("last_step", e["step"])
    arrivals = {}
    for e in events:
        rel = e if e.get("event") == "arrived" else e.get("released")
        if rel and rel["event"] == "arrived":
            arrivals[rel["goal_id"]] = e["step"]
    assert len(trajectory) == 301 and all(t["step"] == i for i, t in enumerate(trajectory))
    encountered = {v: set() for v in THRESHOLDS}
    simultaneous = {v: set() for v in THRESHOLDS}
    first_clear = {}
    occupancy, patches = [], {}
    for i, state in enumerate(trajectory):
        cam = hs.AgentState()
        cam.position = np.asarray(state["xyz"], np.float32)
        cam.rotation = quat_from_coeffs(np.asarray(state["quat_xyzw"],np.float64))
        sim.get_agent(0).set_state(cam, reset_sensors=True)
        mask = np.asarray(sim.get_sensor_observations()["instance_mask"])
        assert mask.shape == (192,192)
        counts = np.bincount(mask.reshape(-1).clip(min=0))
        frame = {t: int(counts[t]) for t in targets if t < len(counts) and counts[t]}
        occupancy.append(frame)
        for t, pix in frame.items():
            for threshold in THRESHOLDS:
                if pix >= math.ceil(threshold * PIXELS):
                    encountered[threshold].add(t)
                    if t in proximity[i]:
                        simultaneous[threshold].add(t)
            if pix >= math.ceil(.01 * PIXELS):
                first_clear.setdefault(t,i)
        if i in starts:
            e = starts[i]
            l, t, r, b = BOXES[e["gpt_patch"]]
            sub = mask[int(t*192):round(b*192), int(l*192):round(r*192)]
            patch = Counter(int(x) for x in sub.ravel() if int(x) in targets)
            patches[i] = patch
    details = []
    for start, e in starts.items():
        gid = e["goal_id"]
        end = arrivals.get(gid, resumed[gid])
        patch = patches[start]
        # Deterministic association to the accepted instance with most pixels inside
        # the nominated patch. A background-only patch has no instance association.
        subject = max(patch, key=lambda sid: (patch[sid],-sid)) if patch else None
        def area(step,t):
            return round(100*occupancy[step].get(t,0)/PIXELS, 3)
        assert trajectory[start-3]["action"]=="turn_right" and trajectory[start-1]["action"]=="turn_left"
        scan_areas = [round(100*max(occupancy[s].values(),default=0)/PIXELS,3) for s in (start-3,start-4,start-1)]
        details.append(dict(
            arrived=gid in arrivals, full_frame_percent_of_patch_target_before=area(start,subject) if subject else None,
            right_center_left_max_target_frame_percent_at_trigger=scan_areas,
            full_frame_percent_of_same_target_after=area(end,subject) if subject else None,
            patch_target_pixels=patch.get(subject,0),
            target_previously_visible_at_one_percent=first_clear.get(subject,301) < start if subject else None,
            new_one_percent_instances_during_interval=sum(start < at <= end for at in first_clear.values()),
            max_any_target_frame_percent_before=round(100*max(occupancy[start].values(),default=0)/PIXELS,3),
            max_any_target_frame_percent_after=round(100*max(occupancy[end].values(),default=0)/PIXELS,3)))
    return dict(distinct_at_frame_fraction={str(t):len(v) for t,v in encountered.items()},
                distinct_close_and_visible_in_same_frame={str(t):len(v) for t,v in simultaneous.items()},
                inspections=details)


def review(i, arm):
    panel = json.loads((ROOT/"local_target_20261128/eval_family/panel.json").read_text())
    spec = panel[i]
    with gzip.open(DATA/"eval/hssdov/content"/(spec["scene"]+".json.gz")) as f:
        ep = next(r for r in json.load(f)["episodes"] if str(r["episode_id"]) == str(spec["id"]))
    private = HERE/("private_events" if arm == "committed" else "private_previous")
    decisions = HERE/"private_events" if arm=="committed" else ROOT/"gpt_inspect_only_20261128/private_events"
    trajectory = lines(private/f"{i}.poses.jsonl")
    classes = _load_goal_sets(DATA)["scoring_class_ids"]["hssdov"][ep["object_category"]]
    records = [r for r in load_scene_instances(DATA/"protocol/instance_viewpoints/hssd","val",ep["scene_id"])
               if r["class_id"] in classes and r["view_points"]]
    proximity = reference_proximity(ep,trajectory,records)
    expected = json.loads((HERE/"accepted.json").read_text())["by_house"][i]
    assert len(set().union(*proximity)) == expected["committed" if arm=="committed" else "inspect_only"]["distinct_approached"]
    sim = configured_sim(ep)
    try:
        lut = instance_lut(sim,"hssd",DATA/"vocab")
        manager = sim.get_rigid_object_manager()
        objects = {obj.semantic_id:obj for obj in (manager.get_object_by_handle(h) for h in manager.get_object_handles())}
        targets = set()
        for row in records:
            sid = row["semantic_id"]
            assert lut[sid] == row["class_id"]+1
            assert Path(objects[sid].creation_attributes.handle).name.removesuffix(".object_config.json") == row["template_name"]
            targets.add(sid)
        assert len(targets)>=2
        report = summary_for_arm(sim, targets, trajectory, decisions/f"{i}.jsonl", proximity)
        return dict(goal=spec["goal"], **report)
    finally:
        sim.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm",choices=["previous","committed"])
    args=p.parse_args()
    result=[review(i,args.arm) for i in range(8)]
    report=dict(arm=args.arm, frames=2408, diagnostic="same center RGB camera geometry, instance mask offline only",
                by_house=result, totals={str(t):sum(r["distinct_at_frame_fraction"][str(t)] for r in result) for t in THRESHOLDS},
                close_and_visible_totals={str(t):sum(r["distinct_close_and_visible_in_same_frame"][str(t)] for r in result) for t in THRESHOLDS})
    (HERE/f"visibility_{args.arm}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report["totals"], flush=True)
