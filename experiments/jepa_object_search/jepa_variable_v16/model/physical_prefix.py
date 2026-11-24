"""Private held-out replay capturing semantic evidence at every actual action."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main(project, asset, house):
    source = project / "jepa_hidden_v11/hssd"
    sys.path.insert(0, str(source))
    from collect_branches import ACT, WORDS, simulate, semantics, world_state
    folder = source / "test" / house
    output = project / "jepa_variable_v16/model/private_dense"
    output.mkdir(exist_ok=True)
    poses = json.loads((folder / "private_replay.json").read_text())
    poses.sort(key=lambda x: hashlib.sha256(("v16-dense|" + x["origin"]).encode()).digest())
    poses = poses[:5]
    if not poses:
        return
    sim = simulate(asset, house)
    agent, lookup = sim.get_agent(0), semantics(sim, asset)
    try:
        for pose in poses:
            origin = pose["origin"]
            target = output / (origin + ".npz")
            if target.exists():
                continue
            with np.load(folder / (origin + ".npz")) as data:
                actions, words = data["actions"], data["words"]
                future, expected = data["future"], data["y"] > 0
            cats = np.asarray([WORDS.index(w) for w in words])
            visible = np.zeros((len(actions), actions.shape[1], len(cats)), bool)
            for k, plan in enumerate(actions):
                agent.set_state(world_state(pose["position"], pose["yaw"]))
                for t, act in enumerate(plan):
                    agent.act(ACT[act])
                    obs = sim.get_sensor_observations()
                    if (t+1) % 6 == 0:
                        if not np.array_equal(obs["rgb"][:,:,:3], future[k, t//6]):
                            raise ValueError("unaligned replay: " + origin)
                    ids, depth = obs["ids"], obs["depth"]
                    good = (ids > 0) & (ids < len(lookup)) & (depth > 0) & (depth <= 2.)
                    counts = np.bincount(ids[good].astype(int), minlength=len(lookup))
                    visible[k,t] = np.isin(cats, lookup[counts >= 32])
            assert np.array_equal(visible[:, 5::6].any(1), expected)
            pending = target.with_suffix(".tmp.npz")
            np.savez_compressed(pending, origin=origin, house=house, visible=visible, words=words, actions=actions)
            pending.replace(target)
    finally:
        sim.close()
    print(house, len(poses), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
