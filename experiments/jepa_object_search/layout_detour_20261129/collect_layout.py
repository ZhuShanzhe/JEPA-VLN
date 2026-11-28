"""Use the train/calibration rooms only; real curved paths, collision and terminal image labels."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'new_view_20261129'
sys.path.insert(0,str(BASE))
from sample_views import simulator,inventory,ASSETS,camera,SOURCE_H,ACTION
from layout_paths import proposals
from pose_data import cue_inputs

def collect(split,house):
    out=HERE/'private_render'/f'{split}_{house}'
    out.mkdir(parents=True,exist_ok=True)
    dst=out/'labels.json'
    if dst.exists():return
    source=BASE/'private_render'/f'{split}_{house}'
    sim=simulator(house)
    rows=[]
    try:
        inventory(sim,ASSETS)
        ag=sim.get_agent(0)
        for idx in range(16):
            with np.load(source/f'{idx}_sensor.npz') as d:
                obs=camera(sim,d['position'],float(d['yaw']),SOURCE_H,0.)
                oid=int(d['oid'])
                _,anchor,box=cue_inputs(obs['ids']==oid,d['depth'])
            original=ag.get_state()
            prop=proposals(anchor,24)
            rng=np.random.default_rng(501782+idx)
            for j,k in enumerate(rng.choice(len(prop),28,replace=False)):
                plan,heading,bend=prop[k]
                ag.set_state(original,reset_sensors=False,infer_sensor_states=False)
                reached=True
                for a in plan:
                    prev=ag.state.position.copy()
                    hit=ag.act(ACTION[a])
                    if a==1 and (hit or np.linalg.norm((ag.state.position-prev)[[0,2]])<.23):
                        reached=False
                        break
                future=sim.get_sensor_observations() if reached else None
                visible=0;img=''
                if future:
                    visible=int((future['ids']==oid).sum())
                    img=f'{idx}_{j}.jpg'
                    Image.fromarray(future['rgb'][...,:3]).save(out/img,quality=90)
                rows.append(dict(idx=idx,plan=plan,reached=reached,px=visible,image=img,bend=bend))
    finally:sim.close()
    tmp=dst.with_suffix('.partial')
    tmp.write_text(json.dumps(rows))
    tmp.replace(dst)
    print(split,house,len(rows),sum(r['reached'] for r in rows),flush=True)
if __name__=='__main__':
    os.umask(0o077)
    i=int(sys.argv[1]);n=int(sys.argv[2])
    split=json.loads((BASE/'view_design.json').read_text())['cohorts']
    jobs=[(s,h) for s in ('train','val') for h in split[s]]
    for s,h in jobs[i::n]:collect(s,h)
