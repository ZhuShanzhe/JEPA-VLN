"""Collect an anchored, ordered panel of HSSD revisit views, without consulting the matcher."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image


STEPS = (("anchor", 0., 0.), ("bridge", .30, 0.), ("same_side", .72, .15),
         ("opposite_side", -.72, .15), ("closer", 0., .70))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run(project, assets, dest):
    previous = project / "jepa_reid_v20"
    sys.path[:0] = [str(project / "jepa_reid_v21/data"), str(previous / "model"), str(previous / "collect")]
    from build_frames import CLOCKS, annotations, open_sim, read_rows, K
    import collect_views
    from replay_depth import observe
    from habitat_sim.utils.common import quat_rotate_vector
    collect_views.WORDS.update(CLOCKS)

    pools = defaultdict(lambda: defaultdict(list))
    for name in ("data", "clocks"):
        for row in read_rows(previous / name / "manifest.jsonl"):
            if row["split"] == "test" and row["category"] in ("clock", "wall_art"):
                pools[(row["house"], row["category"])][row["instance"]].append(row)
    frozen = defaultdict(list)
    for (house, category), instances in sorted(pools.items()):
        for uid in sorted(instances, key=lambda x: digest(["ordered-revisit-1", x]))[:2]:
            seed = min(instances[uid], key=lambda x: digest([uid, x["camera_xyz"], x["camera_heading_deg"]]))
            frozen[house].append(seed)
    dest.mkdir(parents=True, exist_ok=True)
    fingerprint = digest([[h, r["instance"], r["camera_xyz"], r["camera_heading_deg"], r["pitch_deg"]]
                          for h, rows in sorted(frozen.items()) for r in rows])
    protocol = dict(selection_sha256=fingerprint, steps=STEPS, house_cap_per_category=2, pose_min_m=.10)
    frozen_file = dest / "protocol.json"
    if frozen_file.exists() and json.loads(frozen_file.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError("collection protocol differs")
    frozen_file.write_text(json.dumps(protocol, indent=2))
    frames, skipped = [], []
    for house, seeds in sorted(frozen.items()):
        shard = dest / (house + ".json")
        if shard.exists():
            data = json.loads(shard.read_text())
            frames.extend(data["frames"])
            skipped.extend(data["skipped"])
            continue
        sim = open_sim(assets, house)
        part, missing = [], []
        try:
            classes = defaultdict(dict)
            for subtype, objects in collect_views.inventory(sim, assets).items():
                cat = "clock" if subtype in CLOCKS else subtype
                for sid, asset, center in objects:
                    classes[cat][sid] = dict(instance=f"{house}:{sid}", asset=asset, world_center=center.tolist())
            for seed in seeds:
                uid, cat = seed["instance"], seed["category"]
                group = digest([house, uid])[:16]
                index = classes[cat]
                target = np.asarray(index[int(uid.split(":")[-1])]["world_center"])
                old_eye = np.asarray(seed["camera_xyz"])
                old = (old_eye - [0., 1.25, 0.]).astype("float32")
                direction = target - old_eye
                direction[1] = 0.
                direction /= np.linalg.norm(direction)
                sideways = np.array([-direction[2], 0., direction[0]])
                used = []
                for stage, lateral, forward in STEPS:
                    request = (old + lateral * sideways + forward * direction).astype("float32")
                    position = old if stage == "anchor" else np.asarray(sim.pathfinder.try_step(old, request))
                    if not np.isfinite(position).all() or any(np.linalg.norm(position - p) < .10 for p in used):
                        missing.append(dict(group=group, stage=stage, reason="unavailable_distinct_pose"))
                        continue
                    used.append(position.copy())
                    eye = old_eye if stage == "anchor" else position + [0., 1.25, 0.]
                    d = target - eye
                    pose = dict(seed) if stage == "anchor" else dict(seed, camera_xyz=eye.tolist(),
                        camera_heading_deg=math.degrees(math.atan2(d[2], d[0])),
                        pitch_deg=math.degrees(math.atan2(d[1], math.hypot(d[0], d[2]))))
                    obs, state = observe(sim, pose)
                    boxes = annotations(obs["ids"], index)
                    ref = next((b for b in boxes if b["instance"] == uid), None)
                    if stage == "anchor" and (ref is None or ref["xyxy"] != seed["bbox"]):
                        raise ValueError("source registration does not reproduce the original box")
                    name = f"{group}_{stage}"
                    folder = dest / "observations"
                    folder.mkdir(exist_ok=True)
                    Image.fromarray(obs["rgb"][:, :, :3]).save(folder / (name+".png"))
                    np.savez_compressed(folder / (name+".npz"), depth=obs["depth"].astype("float32"),
                                        semantic=obs["ids"].astype("int32"))
                    rotation = np.column_stack([quat_rotate_vector(state.rotation, v) for v in np.eye(3)])
                    transform = np.eye(4)
                    transform[:3,:3] = rotation @ np.diag([1., -1., -1.])
                    transform[:3,3] = state.position
                    part.append(dict(group=group, house=house, category=cat, uid=uid, stage=stage, boxes=boxes,
                                     name=name, K=K, T_world_from_camera=transform.tolist(),
                                     movement_m=float(np.linalg.norm(position-old))))
        finally:
            sim.close()
        temp = shard.with_suffix(".tmp")
        temp.write_text(json.dumps(dict(frames=part, skipped=missing), allow_nan=False))
        temp.replace(shard)
        frames.extend(part)
        skipped.extend(missing)
        print("collected", len(part), len(missing), flush=True)
    assert len(frames) + len(skipped) == sum(map(len, frozen.values())) * len(STEPS)
    assert len({r["name"] for r in frames}) == len(frames)
    (dest/"frames.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in frames))
    (dest/"skipped.json").write_text(json.dumps(skipped))
    print("poses", len(frames), "skipped", len(skipped), "categories", dict(Counter(r["category"] for r in frames)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for key in ("project","assets","out"):
        ap.add_argument(key, type=Path)
    a = ap.parse_args()
    run(a.project,a.assets,a.out)
