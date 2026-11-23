"""Collect paired close and offset poses for active report verification in Habitat."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import habitat_sim as hs
from habitat_sim.utils.common import quat_from_coeffs
import numpy as np


def seed(key):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def save_npz(path, **arrays):
    temp = path.with_suffix(".pending.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def collect(project, root, house, split):
    old = Path(os.environ.get("V8_TEACHER_DIR", project / "jepa_separation_v8_20261123/primitives"))
    sys.path.insert(0, str(old))
    renderer = os.environ.get("V8_RENDERER_DIR", project / "mapping_runtime/pilots/jepa_round2_five_20261121")
    sys.path.insert(0, str(renderer))
    from collect_primitives import OFFICIAL, scene_targets
    from opportunity_labels import ReportViewpoints
    from render_labels import simulation
    ext = project / "jepa_separation_v7_20261123/viewpoint_teacher/exported"
    vocab = list(OFFICIAL.values()) + sorted(json.loads((ext / "query_map.json").read_bytes()).values())
    sim = simulation(house)
    try:
        goals, classes, _, dropped = scene_targets(sim, project / "a_exploration/official/hssd_v025/train/content",
                                                 ext, house, vocab)
        names = np.asarray([vocab[i] for i in classes])
        ids = np.asarray([g["object_id"] for g in goals], dtype=int)
        teacher = ReportViewpoints(sim.pathfinder, goals)
        planned = []
        rng = np.random.default_rng(seed(house + ":v9-view-1"))
        for word in sorted(set(names)):
            indices = sorted(np.flatnonzero(names == word), key=lambda i: seed(house + str(ids[i])))
            for i in indices[:2]:
                vp = goals[i]["view_points"]
                pos = np.array(vp[rng.integers(len(vp))]["agent_state"]["position"], dtype=np.float32)
                pos = np.array(sim.pathfinder.snap_point(pos))
                ang = rng.uniform(0, 2 * math.pi)
                distance = rng.uniform(.7, 1.7)
                offset = np.asarray(sim.pathfinder.snap_point(pos + distance * np.array([math.cos(ang), 0, math.sin(ang)])))
                planned.append(("viewpoint", pos))
                if np.isfinite(offset).all() and abs(offset[1]-pos[1]) <= .35 and np.linalg.norm(offset-pos) >= .4:
                    planned.append(("offset", offset))
        starts = (project / "a_exploration/paired_branches_v2" /
                  ("train" if split == "train" else "dev") / house)
        if split == "holdout":
            starts = project / "jepa_separation_v8_20261123/natural_holdout/raw/dev" / house
        for f in sorted(starts.glob("*/start*/manifest.json"))[:6]:
            planned.append(("natural", np.asarray(json.loads(f.read_bytes())["frozen_pose_xyz_xyzw"][:3])))
        seen = set()
        grid = np.arange(256) * 7 // 256
        cell = grid[:, None] * 7 + grid[None, :]
        lookup = np.full(max(ids) + 1, len(ids), dtype=int)
        lookup[ids] = np.arange(len(ids))
        for kind in ["rows", "observed", "labels"]:
            (root / kind / split / house).mkdir(parents=True, exist_ok=True)
        for provenance, pos in planned:
            xyz = tuple(np.round(pos, 3))
            if xyz in seen:
                continue
            seen.add(xyz)
            sid = f"{split}/{house}/v{len(seen):03d}"
            meta = root / "rows" / (sid + ".json")
            if meta.exists():
                continue
            yaw = np.random.default_rng(seed(sid)).uniform(-math.pi, math.pi)
            st = hs.AgentState()
            st.position = pos.astype(np.float32)
            st.rotation = quat_from_coeffs([0., math.sin(yaw/2), 0., math.cos(yaw/2)])
            agent = sim.get_agent(0)
            agent.set_state(st)
            rgb, masks, near = [], [], []
            for _ in range(12):
                obs = sim.get_sensor_observations()
                instance = obs["semantic"]
                index = lookup[np.where((instance >= 0) & (instance <= max(ids)), instance, 0)]
                valid = index < len(ids)
                flat = index * 49 + cell
                def count(mask):
                    return np.bincount(flat[mask], minlength=len(ids)*49).reshape(len(ids), 7, 7).astype(np.uint16)
                rgb.append(obs["rgb"][:, :, :3].copy())
                masks.append(count(valid))
                near.append(count(valid & (obs["depth"] > 0) & (obs["depth"] <= 2)))
                agent.act("turn_left")
            assert np.linalg.norm(np.asarray(agent.get_state().position)-pos) < 1e-3
            source = root / "observed" / (sid + ".npz")
            label = root / "labels" / (sid + ".npz")
            save_npz(source, rgb=rgb, heading_deg=np.arange(12)*30)
            save_npz(label, grid=masks, near_grid=near, eligible=teacher.eligible(pos), class_name=names,
                     instance_id=ids, pose_xyz=pos, yaw=yaw, provenance=provenance, excluded_classes=np.asarray(dropped,dtype=str))
            temp = meta.with_suffix(".pending")
            temp.write_text(json.dumps(dict(sid=sid,house=house,split=split,observed=str(source),labels=str(label))))
            temp.replace(meta)
        print(house, len(seen), flush=True)
    finally:
        sim.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("project", type=Path)
    p.add_argument("root", type=Path)
    p.add_argument("house")
    p.add_argument("split")
    a = p.parse_args()
    collect(a.project, a.root, a.house, a.split)
