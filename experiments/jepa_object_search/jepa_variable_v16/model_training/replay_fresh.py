"""Measure real per-action instance visibility on the held-out fresh2 houses."""
import json
from pathlib import Path
import sys

import numpy as np


def main(project, asset, house):
    raw = project / "jepa_route_v12/expansion/hssd/test" / house
    sys.path.insert(0, str(project / "jepa_hidden_v11/hssd"))
    from collect_branches import ACT, WORDS, simulate, semantics, world_state
    out = project / "jepa_variable_v16/model_training/private_fresh"
    out.mkdir(exist_ok=True)
    entries = json.loads((raw / "private_replay.json").read_text())
    sim = simulate(asset, house)
    agent, lut = sim.get_agent(0), semantics(sim, asset)
    try:
        for pose in entries:
            key = pose["origin"]
            dst = out / (key + ".npz")
            if dst.exists():
                continue
            with np.load(raw / (key + ".npz")) as a:
                plans, rgb, words, expected = a["actions"], a["future"], a["words"], a["y"] > 0
            cats = [WORDS.index(w) for w in words]
            visible = np.zeros((len(words),len(plans),24),bool)
            for i, plan in enumerate(plans):
                agent.set_state(world_state(pose["position"],pose["yaw"]))
                for t, action in enumerate(plan):
                    agent.act(ACT[action])
                    obs = sim.get_sensor_observations()
                    if t%6 == 5 and not np.array_equal(obs["rgb"][:,:,:3], rgb[i,t//6]):
                        raise ValueError(f"archived RGB disagrees with replay for {key}")
                    seg, depth = obs["ids"], obs["depth"]
                    good = (seg > 0) & (seg < len(lut)) & (depth > 0) & (depth <= 2.)
                    counts = np.bincount(seg[good].astype(int),minlength=len(lut))
                    visible[:,i,t] = np.isin(cats, lut[counts>=32])
            assert np.array_equal(visible[:,:,5::6].any(-1).T, expected)
            tmp = dst.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, visible=visible, words=words, house=house)
            tmp.replace(dst)
    finally:
        sim.close()
    print(house,len(entries),flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])
