"""Revisit all eligible native frontiers on a frozen five-episode historical panel."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import numpy as np


WORDS = {1: "clock.n.01", 14: "wall_art.n.01", 21: "mirror.n.01", 39: "lamp.n.01"}


def read(p):
    return json.loads(p.read_text())


def candidates(row):
    original = row.get("original", "")
    if not original.startswith("f") or not original[1:].isdigit():
        return []
    ref = tuple(a - 1 for a in row.get("original_nominal_actions", []))
    if len(ref) != 24 or not set(ref) <= {0, 1, 2}:
        return []
    found = [ref]
    for label, seq in sorted(row.get("supported_plans", {}).items()):
        if not label.startswith("f") or not label[1:].isdigit() or label == original:
            continue
        seq = tuple(seq)
        if len(seq) == 24 and set(seq) <= {0, 1, 2} and seq not in found:
            found.append(seq)
    return found[:3] if len(found) >= 3 else []


def capture(env, frame, item, rows, request):
    from evaluation.main.evaluator import _env_action, _near_target_ids
    t = request["step"]
    assert t + 28 <= 300
    def near():
        loc = tuple(float(v) for v in env.sim.get_agent_state().position)
        return set(_near_target_ids(loc, item.target_object_ids, item.target_positions, threshold=.25,
                                    pathfinder=env.sim.pathfinder, target_view_points=item.view_points))
    previous = set()
    for n, row in enumerate(rows[:t]):
        pos = np.array(env.sim.get_agent_state().position)
        assert np.linalg.norm(pos - row["position_before"]) < .002
        previous.update(near())
        if row["action_role"] == "navigate":
            frame = env.step(_env_action(row["action"]))
    initial = env.sim.get_agent_state()
    assert np.linalg.norm(initial.position - rows[t]["position_before"]) < .002
    previous.update(near())
    frames = []
    for i, a in enumerate((3, 2, 2, 3)):
        frame = env.step(_env_action(a))
        if i < 3:
            frames.append(np.asarray(frame["rgb"])[..., :3].copy())
    st = env.sim.get_agent_state()
    assert np.linalg.norm(st.position - initial.position) < .002
    assert abs(st.rotation.components @ initial.rotation.components) > .99999
    reached = set()
    visible = set()
    collision = 0
    forward = 0
    for a in request["plan"]:
        frame = env.step(_env_action(int(a) + 1))
        forward += (a == 0)
        collision += (a == 0 and bool(env.sim.previous_step_collided))
        reached.update(near())
        counts = np.bincount(np.maximum(0, np.asarray(frame["semantic"]).ravel().astype(np.int32)))
        visible.update(int(k) for k in item.target_object_ids if k < len(counts) and counts[k] >= 103)
    details = dict(new_reachable=len(reached - previous), reachable=len(reached),
                   target_visible_103px=len(visible), forward=forward, collisions=collision)
    return np.asarray(frames), details


def collect(project, dest):
    base = project / "mapping_runtime"
    panel = base / "pilots/jepa_round10_fixed_five_20261124"
    srv = base / "runs" / read(panel / "mapper_process.json")["run"]
    module_path = base / "analysis/visibility/visibility_replay.py"
    spec = importlib.util.spec_from_file_location("visibility", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    source = inspect.getsource(mod.replay).split("        for r, t in zip(rows, trace):")[0]
    assert "        frame = _reset_environment" in source
    mark = "        cfg.habitat.simulator.scene = item.scene_id"
    assert source.count(mark) == 1
    source = source.replace(mark, mark + "\n        cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = -1")
    source += "        return capture(env,frame,item,rows,out)\n    finally:\n        env.close()\n"
    ns = {**vars(mod), "capture": capture}
    exec(compile(source, "<native MOST episode replay>", "exec"), ns)
    dest.mkdir(parents=True, exist_ok=True)
    events = []
    all_counts = {}
    for case in read(panel / "panel.json")["rows"]:
        ordinal, eid = case["ordinal"], case["episode_id"]
        rec = read(panel / "comparison/accepted/second" / (eid + ".json"))
        result = Path(rec["result"])
        assert hashlib.sha256(result.read_bytes()).hexdigest() == rec["result_sha256"]
        logs = [json.loads(s) for s in (result.parent / "agent/jepa_selection.jsonl").read_text().splitlines()]
        all_counts[str(ordinal)] = {"logged": len(logs), "eligible": 0}
        for row in logs:
            plans = candidates(row)
            t = row["step"]
            if not plans or t + 28 > 300:
                continue
            all_counts[str(ordinal)]["eligible"] += 1
            assert ordinal in WORDS
            tag = f"{ordinal}_{t}"
            file = dest / (tag + ".npz")
            receipt = dest / (tag + ".json")
            if not receipt.is_file():
                frames = None
                scores = []
                for plan in plans:
                    obs, metrics = ns["replay"](result, srv, {"step": t, "plan": plan})
                    if frames is None:
                        frames = obs
                    else:
                        assert np.array_equal(frames, obs)
                    scores.append(metrics)
                np.savez_compressed(file, rgb=frames, plans=plans)
                receipt.write_text(json.dumps(dict(ordinal=ordinal, step=t, outcomes=scores)))
            events.append(tag)
    (dest / "inventory.json").write_text(json.dumps(dict(events=events, counts=all_counts)))


def score(project, dest):
    sys.path.insert(0, str(project / "jepa_env_v13/online"))
    sys.path.insert(0, str(project / "jepa_env_v13/most_integration"))
    from live_search import EnvironmentSearch
    from v13_inputs import camera
    ctrl = EnvironmentSearch(project, WORDS[1], seed=11, horizon=True, layout=False)
    inv = read(dest / "inventory.json")
    cases = []
    panel = project / "mapping_runtime/pilots/jepa_round10_fixed_five_20261124"
    strata = {c["ordinal"]: c["episode_id"] for c in read(panel / "panel.json")["rows"]}
    for tag in inv["events"]:
        info = read(dest / (tag + ".json"))
        eid = strata[info["ordinal"]]
        ref = read(panel / "comparison/accepted/second" / (eid + ".json"))
        ep = read(Path(ref["result"]))["episodes"][0]
        prefix = ep["paper"]["steps"][:info["step"]]
        assert all(a["action"] in (0, 1, 2, 3, 6) for a in prefix)
        assert ep["sensor_hfov"] == 120
        ctrl.reset(WORDS[info["ordinal"]])
        with np.load(dest / (tag + ".npz")) as z:
            native_views = z["rgb"]
            original_fov = ctrl.choose_after_scan(native_views, z["plans"])["index"]
            ctrl.reset(WORDS[info["ordinal"]])
            # No lookup/down in the recorded prefix. Same calibrated 90-degree
            # camera interface as the paid host, using only observed RGB rays.
            adapted = np.stack([camera(frame, 0) for frame in native_views])
            chosen = ctrl.choose_after_scan(adapted, z["plans"])["index"]
        row = {**info, "chosen": chosen, "full_fov_choice": original_fov, "native": 0}
        cases.append(row)
    (dest / "summary.json").write_text(json.dumps(dict(
        protocol="post-hoc cold-start, native frontier candidates, counted physical scan, no paid calls",
        counts=inv["counts"], decisions=cases), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("collect", "score"))
    ap.add_argument("project", type=Path)
    ap.add_argument("cache", type=Path)
    a = ap.parse_args()
    (collect if a.stage == "collect" else score)(a.project, a.cache)
