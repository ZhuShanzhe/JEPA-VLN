"""Collect private RGB training patches with surface-only supervision on the HSSD training split."""
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from local_vision import BOXES


def main(project, house):
    sys.path[:0] = [str(project/"jepa_hidden_v11/hssd"),str(project/"prior_approach_20261128")]
    from collect_branches import simulate, world_state
    from placement_prior import categories, CONTEXTS
    allowed = json.loads((project/"jepa_env_v13/simulator/training_cohort.json").read_text())["train"]
    assert house in allowed
    asset = project/"simulation/hssd-render"
    metadata = project/"simulation/hssd/metadata/fpmodels-with-decomposed.csv"
    with metadata.open() as f:
        vocab = {r["id"]:categories(r) for r in csv.DictReader(f)}
    seed = int(hashlib.sha256(("local-surface-v2|"+house).encode()).hexdigest()[:8],16)%2147483647
    rng = np.random.default_rng(seed)
    sim = simulate(asset,house)
    sim.pathfinder.seed(seed)
    raw = Path(__file__).resolve().parent/"training_private"
    raw.mkdir(exist_ok=True)
    dest = raw/(house+".npz")
    try:
        labels = [10]
        structural = [True]
        furniture = [[] for _ in range(9)]
        objs = sorted(sim.get_rigid_object_manager().get_objects_by_handle_substring("").items())
        for i,(name,obj) in enumerate(objs,1):
            words=next((vocab[k] for k in re.findall(r"[a-fA-F0-9]{40}",name) if k in vocab),set())
            label=next((j for j in range(1,9) if CONTEXTS[j] in words),10)
            labels.append(label)
            structural.append(bool(words & {"wall","floor","ceiling","building","architecture"}) or
                              any(k in name.lower() for k in ("wall","floor","ceiling","stage")))
            obj.semantic_id=i
            if label<9:
                furniture[label].append(np.array(obj.translation))
        poses = []
        for _ in range(16):
            p = sim.pathfinder.get_random_navigable_point()
            if np.all(np.isfinite(p)):
                start = rng.uniform(-math.pi,math.pi)
                poses.extend([(p,start+j*math.pi/2) for j in range(4)])
        for locs in furniture[1:]:
            for idx in rng.permutation(len(locs))[:5]:
                obj = locs[idx]
                for _ in range(8):
                    p=np.asarray(sim.pathfinder.get_random_navigable_point_near(obj,2.6))
                    dx,dz=obj[[0,2]]-p[[0,2]]
                    if np.isfinite(p).all() and .75<np.hypot(dx,dz)<2.55 and abs(p[1]-obj[1])<2.:
                        poses.append((p,math.atan2(-dx,-dz)));break
        labels=np.array(labels,dtype=np.uint8);structural=np.array(structural)
        crops, ys=[],[]
        for pos,yaw in poses:
            sim.get_agent(0).set_state(world_state(pos,yaw))
            obs=sim.get_sensor_observations()
            rgb=obs["rgb"][:,:,:3];depth=obs["depth"]
            h,w=depth.shape
            valid=np.isfinite(depth)&(depth>.15)&(depth<8.)
            yy,xx=np.indices((h,w)); focal=w/2.
            xyz=np.stack([(xx+.5-w/2)*depth/focal,(h/2-yy-.5)*depth/focal,depth],axis=-1)
            n=np.cross(np.gradient(xyz,axis=0), np.gradient(xyz,axis=1))
            norm=np.linalg.norm(n,axis=-1)
            up=np.divide(abs(n[:,:,1]),norm,out=np.ones_like(norm),where=norm>1e-8)
            ids=obs["ids"].astype(int)
            good=(ids>=0)&(ids<len(labels))
            ids=np.where(good,ids,0)
            ymap=labels[ids].copy()
            geometry=good&structural[ids]&valid
            ymap[geometry&(up<.15)]=0
            ymap[geometry&(up>.9)&(abs(xyz[:,:,1]+1.25)<.2)]=9
            for l,t,r,b in BOXES:
                # Class of the visible patch center is what the depth projector anchors.
                u,v=(l+r)/2,(t+b)/2
                center=ymap[int((v-.06)*h):round((v+.06)*h),int((u-.06)*w):round((u+.06)*w)]
                cnt=np.bincount(center.ravel(),minlength=11)
                if cnt.max()<.60*center.size:
                    continue
                patch=rgb[int(t*h):round(b*h),int(l*w):round(r*w)]
                crops.append(np.array(Image.fromarray(patch).resize((112,112),Image.BILINEAR)))
                ys.append(cnt.argmax())
        np.savez_compressed(dest,crops=np.array(crops),labels=np.array(ys,dtype=np.uint8))
        print(house,len(poses),np.bincount(ys,minlength=11).tolist(),flush=True)
    finally:
        sim.close()


if __name__=="__main__":
    main(Path(sys.argv[1]),sys.argv[2])
