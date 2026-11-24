"""Replay a house and label the first sampled view that reveals each target."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def collect(project, asset, split, house, raw_parent=None):
    old = project / "jepa_hidden_v11" / "hssd"
    sys.path.insert(0, str(old))
    from collect_branches import ACT, WORDS, simulate, semantics, world_state

    raw = raw_parent/house if raw_parent else old/split/house
    out = project/"jepa_route_v12"/"data"/"region_masks"/split/house
    out.mkdir(parents=True, exist_ok=True)
    poses = json.loads((raw/"private_replay.json").read_text())
    if not poses:
        (out/"done.json").write_text(json.dumps({"origins": 0}))
        return
    sim = simulate(asset,house)
    agent = sim.get_agent(0)
    lookup = semantics(sim,asset)
    try:
        for pose in poses:
            uid = pose["origin"]
            dst = out/(uid+".npz")
            if dst.exists():
                continue
            with np.load(raw/(uid+".npz")) as d:
                words, actions = d["words"], d["actions"]
                expected, future = d["y"],d["future"]
            indices = [WORDS.index(w) for w in words]
            base = world_state(pose["position"],pose["yaw"])
            tap = np.full(expected.shape,-1,np.int8)
            first_id = np.zeros(expected.shape,np.int32)
            masks = np.zeros((*expected.shape,128,128),bool)
            all_found = [[set() for _ in words] for _ in actions]
            for route,plan in enumerate(actions):
                agent.set_state(base)
                for t,act in enumerate(plan,1):
                    agent.act(ACT[act])
                    if t % 6:
                        continue
                    frame = t//6-1
                    obs = sim.get_sensor_observations()
                    assert np.array_equal(obs["rgb"][:,:,:3], future[route,frame])
                    seg, depth = obs["ids"], obs["depth"]
                    valid = (seg>0)&(seg<len(lookup))&(depth>0)&(depth<=2.)
                    counts = np.bincount(seg[valid], minlength=len(lookup))
                    for q,cat in enumerate(indices):
                        ids = np.flatnonzero((lookup==cat)&(counts>=32))
                        if not len(ids):
                            continue
                        all_found[route][q].update(ids.tolist())
                        if tap[route,q]>=0:
                            continue
                        selected = int(ids[np.argmax(counts[ids])])
                        tap[route,q], first_id[route,q] = frame,selected
                        masks[route,q] = valid & (seg==selected)
            nmax = int(expected.max())
            hit_ids = np.zeros((*expected.shape,nmax),np.int32)
            for route,q in np.ndindex(expected.shape):
                values = sorted(all_found[route][q])
                assert len(values)==int(expected[route,q])
                hit_ids[route,q,:len(values)] = values
            assert np.array_equal(tap>=0, expected>0)
            pairs = [(a,b) for a,b in itertools.combinations(range(len(words)),2)
                     if not np.any((expected[:,a]>0)&(expected[:,b]>0))]
            part = dst.with_suffix(".partial.npz")
            np.savez_compressed(part, first_tap=tap, first_id=first_id, hit_ids=hit_ids,
                                pack_mask=np.packbits(masks,axis=-1),
                                pair_queries=np.asarray(pairs,np.int32).reshape(-1,2),
                                y=expected, words=words, origin=uid, house=house)
            part.replace(dst)
        (out/"done.json").write_text(json.dumps({"origins":len(poses)}))
    finally:
        sim.close()


def main(a):
    if a.house:
        return collect(a.project,a.asset,a.split,a.house,a.raw_parent)
    old=a.project/"jepa_hidden_v11/hssd"
    cohort=json.loads((old/"cohort.json").read_text())
    jobs=[(s,h) for s in ("train","calibration") for h in cohort[s]]
    root=a.project/"jepa_route_v12/data/logs"
    root.mkdir(exist_ok=True,parents=True)
    def run(job):
        split,house=job
        with (root/(house+".log")).open("a") as log:
            subprocess.run([sys.executable,"-Bu",__file__,str(a.project),str(a.asset),"--split",split,"--house",house],
                           env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MAGNUM_LOG="quiet",HABITAT_SIM_LOG="quiet"),
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        print(split,house,flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(run,jobs))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("project",type=Path)
    p.add_argument("asset",type=Path)
    p.add_argument("--split",choices=["train","calibration","extra_train"])
    p.add_argument("--raw-parent",type=Path)
    p.add_argument("--house")
    p.add_argument("--workers",type=int,default=3)
    main(p.parse_args())
