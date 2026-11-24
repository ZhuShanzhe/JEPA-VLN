"""Replay previously acquired images, then evaluate frozen choices by past visibility."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def summarize(rows):
    group = defaultdict(list)
    for row in rows:
        group[row["house"]].append(row)
    metric_names = ["online", "fixed", "observed", "no_semantic_memory", "v8", "oracle"]
    metrics = {k: float(np.mean([np.mean([r[k] for r in items]) for items in group.values()]))
               for k in metric_names} if rows else {}
    return dict(conditions=len(rows), houses=len(group), source_origins=len({r["sid"] for r in rows}),
                metrics_house_equal_weight=metrics)


def run(root):
    from habitat_sim import AgentState
    from habitat_sim.utils.common import quat_from_coeffs
    v8 = root / "jepa_separation_v8_20261123"
    v10 = root / "jepa_memory_v10_20261124"
    sys.path[:0] = [str(v8 / "primitives"), str(root / "mapping_runtime/pilots/jepa_round2_five_20261121"),
                    str(v10 / "models")]
    from collect_primitives import NAMES, scene_targets, grid_labels
    from render_labels import simulation
    from v10_live import prior_transform

    weights = json.loads((v10 / "models/v10.json").read_text())["weights"]
    results, frames, worst = defaultdict(list), 0, 0.
    info, zero, positive_ties = defaultdict(int), defaultdict(int), defaultdict(int)
    rows = json.loads((v8 / "primitives/holdout.index.json").read_text())
    for house in sorted({r["house"] for r in rows}):
        sim = simulation(house)
        try:
            for r in (r for r in rows if r["house"] == house):
                uid = hashlib.sha256(r["sid"].encode()).hexdigest()
                with np.load(v10 / "features" / (uid + ".npz")) as f:
                    words, features, gains, present = f["words"].tolist(), f["f"], f["y"], f["present"]
                _, _, lookup, _ = scene_targets(sim, root / "a_exploration/official/hssd_v025/train/content",
                    root / "jepa_separation_v7_20261123/viewpoint_teacher/exported", house, words)
                start = v8 / "natural_holdout/raw/dev" / house / Path(r["sid"]).parent.name / Path(r["sid"]).name
                pose = json.loads((start / "manifest.json").read_text())["frozen_pose_xyz_xyzw"]
                s = AgentState()
                s.position = np.array(pose[:3])
                s.rotation = quat_from_coeffs(np.array(pose[3:]))
                agent = sim.get_agent(0)
                agent.set_state(s)
                max_pixels = np.zeros(len(words), dtype=int)
                with np.load(v10 / "prefixes" / (uid + ".npz")) as pre:
                    frame_at = {int(t): i for i, t in enumerate(pre["frame_steps"])}
                    for t, action in enumerate(pre["actions"]):
                        if t:
                            agent.act(NAMES[int(action)])
                        if t not in frame_at:
                            continue
                        obs = sim.get_sensor_observations()
                        image = obs["rgb"][:, :, :3]
                        diff = np.abs(image.astype("int16") - pre["rgb"][frame_at[t]].astype("int16")).mean()
                        worst = max(worst, float(diff))
                        if diff > 3.0:
                            raise ValueError(f"RGB replay mismatch at frame {t}: {diff}")
                        axis = np.arange(image.shape[0]) * 7 // image.shape[0]
                        grid, _ = grid_labels(obs, lookup, len(words), 7 * axis[:, None] + axis[None, :])
                        max_pixels = np.maximum(max_pixels, grid.max(axis=(-1, -2)))
                        frames += 1
                for hi, feature in enumerate(features):
                    p, o, g = (prior_transform(feature[..., i]) for i in (0, 1, 2))
                    novelty, hazard, trace = (feature[..., i] for i in (3, 4, 5))
                    motion = weights["novelty"]*novelty + weights["clearance"]*hazard
                    geo, cue = weights["geometry"]*g, weights["target_trace"]*trace
                    fixed = weights["prior"]*p + geo
                    scores = dict(online=fixed+motion+cue, fixed=fixed, observed=weights["prior"]*o+geo+motion+cue,
                                  no_semantic_memory=fixed+motion, v8=feature[..., 0])
                    for q in np.flatnonzero(present):
                        bucket = "never_visible" if max_pixels[q] == 0 else ("weak_only" if max_pixels[q] < 20 else "clearly_seen")
                        y = gains[hi, :, q]
                        info[bucket] += 1
                        if y.max() == y.min():
                            if y.max() > 0:
                                positive_ties[bucket] += 1
                            else:
                                zero[bucket] += 1
                            continue
                        row = {k: float(y[v[:, q].argmax()]) for k, v in scores.items()}
                        results[bucket].append(dict(row, house=house, sid=uid, oracle=float(y.max())))
        finally:
            sim.close()
    return dict(houses=len({r["house"] for r in rows}), origins=len(rows), matched_rgb_frames=frames,
                max_rgb_mean_absolute_error=worst, all_conditions=dict(info), all_branches_zero=dict(zero),
                all_branches_same_positive=dict(positive_ties),
                informative={k:summarize(results[k]) for k in ("never_visible", "weak_only", "clearly_seen")},
                unit="One goal x origin x horizon. Gains count newly eligible distinct instances. Informative cases have at least two different candidate outcomes.",
                scope="Never visible means zero teacher pixels of the category across all retained RGB keyframes in this executed prefix. Evaluation only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.project_root)
    args.output.write_text(json.dumps(result, indent=2))
    print("complete", result["matched_rgb_frames"], result["all_conditions"])
