"""Create one controlled nearby past camera per frozen HSSD query, without object labels."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("project","assets","out"):
        ap.add_argument(name,type=Path)
    args=ap.parse_args()
    p=args.project.resolve()
    sys.path[:0]=[str(p/"jepa_reid_v21/data"), str(p/"jepa_reid_v20/model")]
    from build_frames import open_sim
    from replay_depth import HEIGHT, observe
    from habitat_sim.utils.common import quat_rotate_vector
    private=p/"identity_stage3_model/new_houses/private"
    def read(name):
        return [json.loads(line) for line in (private/name).read_bytes().splitlines()]
    queries={r["query"] for r in read("cases.jsonl") if r["kind"]=="same"}
    jobs=defaultdict(list)
    for row in read("frames.jsonl"):
        if row["name"] in queries:
            jobs[row["house"]].append(row)
    args.out.mkdir(parents=True,exist_ok=True,mode=0o700)
    for house,group in jobs.items():
        pending=[r for r in group if not (args.out/(r["name"]+".npz")).is_file()]
        if not pending:
            continue
        sim=open_sim(args.assets,house)
        try:
            for row in pending:
                T=np.asarray(row["T"])
                eye=T[:3,3]
                forward=T[:3,2]
                horizontal=forward.copy()
                horizontal[1]=0
                horizontal/=np.linalg.norm(horizontal)
                foot=eye-[0,HEIGHT,0]
                old=np.asarray(sim.pathfinder.try_step(foot,foot-.25*horizontal))
                assert np.isfinite(old).all() and np.linalg.norm(old-foot)<.35
                sign=1 if hashlib.sha256(row["name"].encode()).digest()[0]%2 else -1
                history=dict(camera_xyz=(old+[0,HEIGHT,0]).tolist(),
                    camera_heading_deg=float(np.rad2deg(np.arctan2(forward[2],forward[0])))+10*sign,
                    pitch_deg=float(np.rad2deg(np.arcsin(np.clip(forward[1],-1,1)))))
                obs,state=observe(sim,history)
                pose=np.eye(4)
                pose[:3,:3]=np.column_stack([quat_rotate_vector(state.rotation,v) for v in np.eye(3)])@np.diag([1.,-1.,-1.])
                pose[:3,3]=state.position
                assert np.rad2deg(np.arccos(np.clip((np.trace(pose[:3,:3].T@T[:3,:3])-1)/2,-1,1)))<11
                name=args.out/(row["name"]+".npz")
                temp=name.with_suffix(".partial.npz")
                np.savez_compressed(temp,rgb=obs["rgb"][:,:,:3],depth=obs["depth"],T=pose)
                temp.replace(name)
        finally:
            sim.close()
    assert len(list(args.out.glob("*.npz")))==len(queries)
    print("complete",len(queries),len(jobs))


if __name__=="__main__":
    main()
