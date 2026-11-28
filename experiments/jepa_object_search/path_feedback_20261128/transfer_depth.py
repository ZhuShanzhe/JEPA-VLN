"""Recover beginning-of-plan depth for the frozen 12–108 action holdout. Future labels stay in the evaluator."""
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from io import BytesIO

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
PRIOR=PROJECT/"pose_horizon_20261128"
sys.path.insert(0,str(PRIOR))
from pose_data import simulator,camera,SOURCE_H


def main():
    os.umask(0o077)
    dest=HERE/"private_transfer_depth"
    dest.mkdir(exist_ok=True)
    for shard in sorted((PRIOR/"private_long_cue").glob("test_*.npz")):
        key=shard.stem
        house=key.split("_",1)[1]
        output=dest/(key+".npz")
        if output.exists():continue
        with np.load(shard) as z: origins=np.unique(z["origin"])
        starts=json.loads((PROJECT/"jepa_env_v13/simulator/trees/fresh"/house/"private_states.json").read_text())
        starts={s["origin"]:s for s in starts}
        sim=simulator(house)
        depths=[]
        try:
            for origin in origins:
                s=starts[origin]
                frame=camera(sim,np.asarray(s["position"],np.float32),s["yaw"],SOURCE_H,0.)
                raw=BytesIO(); Image.fromarray(frame["rgb"][...,:3]).save(raw,format="JPEG",quality=92)
                assert raw.getvalue()==(PRIOR/"private_long_cue_render"/key/(origin+"_src.jpg")).read_bytes()
                depths.append(frame["depth"].squeeze().copy())
        finally:sim.close()
        np.savez_compressed(output,origin=origins,depth=np.asarray(depths))
    print("Reconstructed and verified source images for",len(list(dest.glob("*.npz"))),"held-out houses",flush=True)

if __name__=="__main__":main()
