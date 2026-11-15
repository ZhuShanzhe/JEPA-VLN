"""Private, strictly post-episode Habitat replay of the delivered camera path."""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from habitat.config import read_write
from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig
from habitat.utils.geometry_utils import quaternion_from_coeff
from omegaconf import OmegaConf
from evaluation.main import run_eval
from evaluation.main.evaluator import _reset_environment, _env_action, _action_role


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def view_signature(rgb):
    return np.asarray(Image.fromarray(np.asarray(rgb, dtype=np.uint8)[:, :, :3]).resize(
        (64, 64), Image.Resampling.BOX), dtype=float)


def replay(result_path, service_dir, out):
    start = time.monotonic()
    doc = json.loads(result_path.read_text())
    ep, = doc["episodes"]
    opts = doc["manifest"]["arguments"]
    assert opts["goal_type"] == "description" and opts["action_protocol"] == "goat"
    assert ep["target_mode"] == "many" and not ep["timed_out"]
    eid = ep["episode_id"]
    house = Path(ep["scene_id"]).name.removesuffix(".basis.glb")
    group = Path(opts["dataset_dir"]) / house
    data = run_eval.load_custom_dataset(*(str(group / (k + ".json")) for k in ("episodes","queries","goals")),
                                        goal_type="description")
    (item,) = [e for e in data if e.episode_id == eid]
    data.episodes = [item]
    root = Path(opts["scene_root"])
    cfg = run_eval.load_habitat_config(opts["config"])
    run_eval.configure_hm3d_scene_root(cfg, root)
    run_eval.attach_resolved_scene_paths(data, root)
    objects = sorted(map(int, item.target_object_ids))
    assert objects == sorted(ep["paper"]["episode_definition"]["target_object_ids"])
    desc = Path(item.scene_id).with_name(house+".semantic.txt")
    with desc.open(encoding="utf-8") as f:
        assert "HM3D" in next(f)
        categories = {int(r[0]): r[2] for r in csv.reader(f) if len(r) >= 3}
    assert set(objects) <= categories.keys()
    rgb_cfg = cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb
    sem = OmegaConf.structured(HabitatSimSemanticSensorConfig)
    for field in ["hfov", "position", "orientation"]:
        sem[field] = rgb_cfg[field]
    sem.width, sem.height = int(rgb_cfg.width) // 4, int(rgb_cfg.height) // 4
    assert sem.width == sem.height == 320 and int(rgb_cfg.hfov) == ep["sensor_hfov"] == 120
    with read_write(cfg.habitat.simulator):
        cfg.habitat.simulator.agents.main_agent.sim_sensors["semantic"] = sem
        cfg.habitat.simulator.scene = item.scene_id
    rows = ep["paper"]["steps"]
    n = len(rows)
    assert n == ep["steps_taken"] and [r["step"] for r in rows] == list(range(n))
    trace = [json.loads(s) for s in (result_path.parent / "agent/action_trace.jsonl").read_text().splitlines()]
    assert [(r["step"],r["action"]) for r in rows] == [(r["step"],r["actual_action"]["id"]) for r in trace]
    pics = service_dir/"mapping/diagnostics"/(eid+"_frames")
    assert pics.is_dir()
    sample = set(np.linspace(0, n-1, min(12,n), dtype=int))
    positions_error, rgb_mae, masks, observed_poses = [], [], [], []
    controls = {}
    env = run_eval.Env(cfg)
    try:
        frame = _reset_environment(env, item, data)
        annotation_objects = env.sim.semantic_annotations().objects
        # Verify the rendered ID index against the scanned-house descriptor.
        for oid in objects:
            o = annotation_objects[oid]
            assert int(o.id.rsplit("_",1)[-1]) == oid and o.category.name().casefold() == categories[oid].casefold(), (oid, o.id)
        for r, t in zip(rows, trace):
            state = env.sim.get_agent_state()
            diff = float(np.linalg.norm(np.array(state.position)-r["position_before"]))
            assert diff < 0.002, ("pose mismatch", r["step"],diff)
            positions_error.append(diff)
            depthfree = np.asarray(frame["semantic"]).squeeze()
            assert depthfree.shape == (sem.height, sem.width) and np.issubdtype(depthfree.dtype, np.integer)
            counts = np.bincount(np.clip(depthfree.ravel(), 0, len(annotation_objects)),minlength=len(annotation_objects)+1)
            masks.append(counts[objects])
            rgb_state = state.sensor_states["rgb"]
            sem_state = state.sensor_states["semantic"]
            assert np.allclose(rgb_state.position,sem_state.position,atol=1e-6) and abs(
                abs(np.dot(rgb_state.rotation.components,sem_state.rotation.components))-1.0)<1e-5
            observed_poses.append(np.r_[rgb_state.position,rgb_state.rotation.components])
            if r["step"] in sample and t["mapping_frame_id"] is not None:
                picture = pics / ("rgb_%06d.jpg" % t["mapping_frame_id"])
                with Image.open(picture) as img:
                    error = float(np.abs(view_signature(np.asarray(img))-view_signature(frame["rgb"])).mean())
                assert error <= 3.0, ("RGB reference mismatch", r["step"], round(error,3))
                rgb_mae.append(error)
            role = _action_role(r["action"], item.target_mode, opts["action_protocol"])
            assert role == r["action_role"]
            if role == "navigate":
                frame = env.step(_env_action(r["action"]))
            elif role not in ("target_found","finish"):
                raise ValueError("Unexpected role "+role)
        last = ep["trajectory_positions"][-1]
        assert np.linalg.norm(np.asarray(env.sim.get_agent_state().position)-last["position"]) < 0.002
        # Positive controls use annotated camera positions only AFTER the complete replay.
        # They never enter trajectory coverage or the policy.
        goals = json.loads((group/"goals.json").read_text())["goals"]
        selected = {int(g["object_id"]):g for g in goals.values() if int(g["object_id"]) in objects}
        assert set(selected) == set(objects)
        for oid,g in selected.items():
            assert g["category"].strip().casefold() == categories[oid].strip().casefold() or str(g["object_name"]).strip().casefold() == categories[oid].strip().casefold(), "Descriptor/source category mismatch"
            candidates = [v["agent_state"] for v in g["view_points"]]
            maximum = 0
            for idx in np.linspace(0,len(candidates)-1,min(12,len(candidates)),dtype=int):
                v = candidates[idx]
                obs = env.sim.get_observations_at(np.array(v["position"],dtype=np.float32),
                                                  quaternion_from_coeff(v["rotation"]),keep_agent_at_new_pose=False)
                maximum = max(maximum,int(np.count_nonzero(obs["semantic"]==oid)))
                if maximum >= 10:
                    break
            controls[str(oid)] = maximum
    finally:
        env.close()
    assert len(rgb_mae) >= min(n,4)
    vis = np.asarray(masks)
    poses = np.asarray(observed_poses)
    thresholds = {"any":(1,1), "moderate":(103,2), "large":(512,3)}
    targets = []
    for col, oid in enumerate(objects):
        target = {"id":oid, "max_pixels":int(vis[:,col].max()), "exposure":{},
                  "reached":oid in ep["reached_targets"],"approached":oid in ep["approached_target_ids"]}
        for name,(pixels,needed) in thresholds.items():
            visible = vis[:,col] >= pixels
            run = 0
            first = None
            for i,v in enumerate(visible):
                # Exclude repeated stationary observations as evidence of a second view.
                if not v:
                    run = 0
                elif run == 0 or not np.allclose(poses[i], poses[i-1], atol=1e-5,rtol=0):
                    run += 1
                if run >= needed and first is None:
                    first = i
            target["exposure"][name] = {"visible_steps":int(visible.sum()),"first_qualifying_step":first}
        targets.append(target)
    summary = {name:sum(t["exposure"][name]["first_qualifying_step"] is not None for t in targets)
               for name in thresholds}
    decisions = sorted(ep["paper"]["agent_trace"]["decisions"], key=lambda x:x["step"])
    steer = ep["paper"]["agent_trace"]["steering_events"]
    windows=[]
    for di,d in enumerate(decisions):
        command = d.get("output") or {}
        fs = d.get("world_state",{}).get("frontiers",[])
        if d.get("decision_origin") != "vlm" or d.get("validation")!="ok" or command.get("action")!="GOTO_FRONTIER":
            continue
        a, b = int(d["step"]), int(decisions[di+1]["step"]) if di+1<len(decisions) else n-1
        if b<=a or len(fs)<2 or command.get("target_id") not in {f["id"] for f in fs}:
            continue
        if not any(s.get("applied") and s.get("step")==a and s.get("decision_id")==d.get("decision_id")
                   and s.get("requested",{}).get("target_id")==command["target_id"] for s in steer):
            continue
        new={}
        for level in thresholds:
            new[level]=sum(a < t["exposure"][level]["first_qualifying_step"] <= b for t in targets
                           if t["exposure"][level]["first_qualifying_step"] is not None)
        windows.append({"decision_step":a,"last_delivered_obs":b,"candidate_count":len(fs),"new_crossings":new})
    reliable=sum(controls[str(t["id"])]>=10 or t["max_pixels"]>0 for t in targets)
    missing_controls=sum(controls[str(t["id"])]<10 for t in targets)
    no_pixels_no_control=len(targets)-reliable
    result = {"schema":1,"episode_id":eid,"scene":house,"source_result_sha256":digest(result_path),
              "source_script_sha256":digest(Path(__file__)),"steps_replayed":n,"semantic_resolution":[320,320],
              "thresholds":thresholds,"k":ep["required_target_count"],"eligible_instances":len(targets),
              "count_exposed":summary,"targets":targets,"max_position_error_m":max(positions_error),
              "reference_rgb_frames":len(rgb_mae),"max_reference_rgb_mae_at_64":max(rgb_mae),
              "heldout_reference_semantic_pixels":controls,
              "annotation_control_instances_with_10px":sum(v >= 10 for v in controls.values()),
              "reference_control_missing":missing_controls, "never_seen_unconfirmed":no_pixels_no_control,
              "reported_distinct":ep["found_count"],"candidate_windows":windows,
              "elapsed_sec":round(time.monotonic()-start,2)}
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix(".tmp."+str(os.getpid()))
    tmp.write_text(json.dumps(result,allow_nan=False)+"\n")
    try:
        os.link(tmp,out)
    finally:
        tmp.unlink()
    print(json.dumps({k:result[k] for k in ["steps_replayed","count_exposed","max_position_error_m",
                                            "reference_rgb_frames","max_reference_rgb_mae_at_64","elapsed_sec"]}))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("result",type=Path)
    p.add_argument("service",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    replay(a.result,a.service,a.out)
