"""Repeat every physical branch and separately measure horizontal and vertical movement."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import habitat_sim as hs
from habitat_sim.utils.common import quat_from_coeffs
import numpy as np


def replay(data, renderer, base_starts, heldout_starts):
    sys.path.insert(0, str(renderer))
    from render_labels import simulation

    grouped = defaultdict(list)
    for split in ["train", "calibration", "holdout"]:
        for row in json.loads((data / (split + ".index.json")).read_bytes()):
            grouped[(split, row["house"])].append(row)
    result = {}
    for (split, house), rows in grouped.items():
        metrics = result.setdefault(split, dict(states=0, actions=0, collisions=0,
            max_replay_difference_m=0., max_horizontal_step_m=0., max_abs_vertical_step_m=0.,
            max_spatial_step_m=0., steps_3d_over_251mm=0))
        sim = simulation(house)
        try:
            agent = sim.get_agent(0)
            for row in rows:
                path = heldout_starts if split == "holdout" else (base_starts / ("train" if split == "train" else "dev"))
                manifest = path.joinpath(*row["sid"].split("/")[2:]) / "manifest.json"
                pose = json.loads(manifest.read_bytes())["frozen_pose_xyz_xyzw"]
                state = hs.AgentState()
                state.position = pose[:3]
                state.rotation = quat_from_coeffs(pose[3:])
                with np.load(row["observed"]) as o, np.load(row["labels"]) as lab:
                    actions = o["actions"]
                    ref = np.diff(lab["actual_progress"], axis=1, prepend=0)
                    ref_collisions = lab["collisions"]
                for i, route in enumerate(actions):
                    agent.set_state(state)
                    before = np.asarray(agent.get_state().position).copy()
                    for t, action in enumerate(route):
                        collision = agent.act(("move_forward", "turn_left", "turn_right")[action])
                        after = np.asarray(agent.get_state().position).copy()
                        delta = after - before
                        before = after
                        horizontal = float(np.linalg.norm(delta[[0, 2]]))
                        spatial = float(np.linalg.norm(delta))
                        diff = abs(spatial - ref[i, t])
                        assert bool(collision) == bool(ref_collisions[i, t])
                        assert horizontal <= .251 and diff <= 1e-4, (house, row["sid"], i, t, horizontal, diff)
                        if action != 0:
                            assert spatial < 1e-5
                        metrics["actions"] += 1
                        metrics["collisions"] += int(collision)
                        metrics["steps_3d_over_251mm"] += int(spatial > .251)
                        for key, value in (("max_replay_difference_m", diff), ("max_horizontal_step_m", horizontal),
                                           ("max_abs_vertical_step_m", abs(float(delta[1]))), ("max_spatial_step_m", spatial)):
                            metrics[key] = max(metrics[key], float(value))
                metrics["states"] += 1
        finally:
            sim.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "renderer", "base_starts", "heldout_starts", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    result = replay(args.data, args.renderer, args.base_starts, args.heldout_starts)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
