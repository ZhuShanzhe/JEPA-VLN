"""Acquire new oblique and side-step RGB-D views of existing training and calibration cues."""
import argparse
import json
import os
from pathlib import Path
import sys
import math
import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'new_view_20261129'
sys.path.insert(0, str(PRIOR))
from sample_views import simulator, inventory, ASSETS, camera, SOURCE_H, ACTION


def acquire(split, house):
    out = HERE/'private_clarity_views'/split/house
    out.mkdir(parents=True, exist_ok=True)
    sim = simulator(house)
    try:
        inventory(sim, ASSETS)
        ag = sim.get_agent(0)
        for path in sorted((PRIOR/'private_render'/(split+'_'+house)).glob('*_sensor.npz')):
            dst = out/(path.name.split('_')[0]+'.npz')
            if dst.exists(): continue
            with np.load(path) as data:
                pos, yaw = data['position'], float(data['yaw'])
                oid, box = int(data['oid']), data['box']
            camera(sim, pos, yaw, SOURCE_H, 0)
            initial = ag.get_state()
            frames, seen = [], set()
            def save():
                s = ag.state
                delta = s.position-pos
                angle = 2*math.atan2(s.rotation.y, s.rotation.w)-yaw
                pose = (math.cos(yaw)*delta[0]-math.sin(yaw)*delta[2],-math.sin(yaw)*delta[0]-math.cos(yaw)*delta[2],math.atan2(math.sin(angle),math.cos(angle)))
                key = tuple(round(v,2) for v in pose)
                if key in seen: return
                seen.add(key)
                obs = sim.get_sensor_observations()
                frames.append((obs['rgb'][...,:3].copy(), obs['depth'].squeeze().astype('float16'), np.packbits(obs['ids']==oid), pose))
            save()
            for sign in (2,3):
                other = 5-sign
                # Two genuinely different lateral displacements with repeated views toward the anchor.
                for steps in (2,5):
                    ag.set_state(initial, reset_sensors=False, infer_sensor_states=False)
                    for a in [sign]+[1]*steps:
                        hit = ag.act(ACTION[a])
                        if hit: break
                    for a in [other,1,1,other,1,1,sign,1,1]:
                        hit = ag.act(ACTION[a]); save()
                        if hit and a==1: break
            rgb,dep,mask,pose = map(np.asarray,zip(*frames))
            temp = dst.with_suffix('.partial.npz')
            np.savez_compressed(temp,rgb=rgb,depth=dep,mask=mask,pose=pose,box=box)
            temp.replace(dst)
    finally:
        sim.close()


if __name__=='__main__':
    os.umask(0o077)
    p = argparse.ArgumentParser();p.add_argument('index',type=int);p.add_argument('workers',type=int);args=p.parse_args()
    cohorts = json.loads((PRIOR/'view_design.json').read_text())['cohorts']
    for split,house in [(s,h) for s in ('train','val') for h in cohorts[s]][args.index::args.workers]:
        acquire(split,house)
