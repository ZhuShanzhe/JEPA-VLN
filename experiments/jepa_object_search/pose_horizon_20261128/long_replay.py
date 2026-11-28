"""Replay exported action sequences from original starting states and check teacher pixels."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
TREES=ROOT/"jepa_env_v13/simulator/trees"


def verify(split,source,asset):
    from collect_branches import simulate, semantics, world_state, ACT
    shard=next(iter(sorted((HERE/"private_long").glob(split+"-*.npz"))))
    house=shard.stem.split("-",1)[1]
    folder=TREES/source/house
    start=json.loads((folder/"private_states.json").read_text())[0]
    origin=start["origin"]
    with np.load(shard) as z:
        mask=z["group"]==f"{origin}:r:h108"
        plans=z["action"][mask].astype(int)-1
    assert plans.shape==(3,108)
    sim=simulate(asset,house)
    pixel_count=0
    physical=0
    try:
        semantics(sim,asset)
        agent=sim.get_agent(0)
        for first,issued in enumerate(plans):
            agent.set_state(world_state(start["position"],start["yaw"]))
            key="r"; cursor=0
            for depth in range(1,5):
                branch=first if depth==1 else int.from_bytes(
                    sha256(f"{origin}/r/{first}/{depth}".encode()).digest()[:4],"big")%3
                with np.load(folder/origin/f"node_{key}.npz") as node:
                    for ix,a in enumerate((2,1,1,2)):
                        if depth!=1:
                            assert issued[cursor]==a
                            cursor+=1
                        agent.act(ACT[a])
                        if ix<3:
                            rgb=sim.get_sensor_observations()["rgb"][:,:,:3]
                            assert np.array_equal(rgb,node["ctx"][ix])
                            pixel_count+=1
                    for t,a in enumerate(node["plans"][branch],1):
                        assert a==issued[cursor]
                        cursor+=1
                        blocked=agent.act(ACT[int(a)])
                        assert bool(a==0 and blocked)==bool(node["collided"][branch,t-1])
                        if t%6==0:
                            rgb=sim.get_sensor_observations()["rgb"][:,:,:3]
                            assert np.array_equal(rgb,node["walk_rgb"][branch,t//6-1])
                            pixel_count+=1
                key+=str(branch)
            assert cursor==len(issued)
            physical+=cursor
    finally:
        sim.close()
    return dict(origins=1,sibling_trajectories=3,actions_after_source=physical,
                exact_rgb_images=pixel_count,mismatched_images=0,contact_mismatches=0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--assets",type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,str(ROOT/"jepa_route_v12/expansion/hssd"))
    result={k:verify(k,v,args.assets) for k,v in (("train","train"),("val","calibration"),("test","fresh"))}
    (HERE/"long_replay.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    main()
