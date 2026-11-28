"""Acquire real yaw views for a predeclared panel: first 5 sampled cues in each held-out building."""
import hashlib
from io import BytesIO
from pathlib import Path
import os
import numpy as np
from PIL import Image
from collect_paths import HERE,ASSETS,SOURCE_H,camera,choose_sources,houses,inventory,simulator,SAMPLES_PER_HOUSE


def collect():
    os.umask(0o077)
    out=HERE/"private_looks"
    out.mkdir(exist_ok=True)
    for house in houses()["test"]:
        destination=out/(house+".npz")
        if destination.exists():continue
        seed=int(hashlib.sha256(("path_feedback:paired:v1:"+house).encode()).hexdigest()[:8],16)%(2**31-1)
        sim=simulator(house);sim.seed(seed);sim.pathfinder.seed(seed)
        frames=[];positions=[];angles=[];instances=[]
        try:
            sources=choose_sources(sim,inventory(sim,ASSETS),np.random.default_rng(seed),SAMPLES_PER_HOUSE)
            agent=sim.get_agent(0)
            for i,(_,sid,p,yaw,_) in enumerate(sources[:5]):
                src=camera(sim,p,yaw,SOURCE_H,0.)
                raw=BytesIO();Image.fromarray(src["rgb"][...,:3]).save(raw,format="JPEG",quality=92)
                assert raw.getvalue()==(HERE/"private_render"/("test_"+house)/f"{i}_source.jpg").read_bytes()
                dep=[src["depth"].squeeze().copy()]
                agent.act("turn_left")
                dep.append(sim.get_sensor_observations()["depth"].squeeze().copy())
                agent.act("turn_right");agent.act("turn_right")
                dep.append(sim.get_sensor_observations()["depth"].squeeze().copy())
                frames.append(dep);positions.append(p);angles.append(yaw);instances.append(sid)
        finally:sim.close()
        np.savez_compressed(destination,depth=np.asarray(frames),position=positions,yaw=angles,sid=instances)
    print("Verified source RGB and collected observation turns in all 12 held-out houses",flush=True)

if __name__=="__main__":collect()
