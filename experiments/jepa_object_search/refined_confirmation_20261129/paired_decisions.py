"""Stop at first acceptance on a stream of measured RGB-D frames. No scoring labels enter this process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent/'reactive_view_20261129'))
from frame_confirm import FrameConfirmer

ARMS = ('old','clarity','memory','full')


def apply_arm(path, destination, arm):
    with np.load(path, allow_pickle=False) as sensors:
        rgb, depth, odo = sensors['rgb'], sensors['depth'], sensors['odom']
        cue, allowance = sensors['cue'], int(sensors['budget'])
    if arm == 'old':
        model = FrameConfirmer(rgb[0], depth[0], cue)
    else:
        from memory_refit import RefinedMemory
        model = RefinedMemory(rgb[0], depth[0], cue, use_memory=arm in ('memory','full'),
                              use_clarity=arm in ('clarity','full'))
    assert len(rgb)-1 <= allowance
    trace = []
    submit = None
    for t,(image,z,pose) in enumerate(zip(rgb,depth,odo)):
        decision = model.observe(image,z,pose)
        box = tuple(map(int,decision['box'])) if decision.get('box') is not None else None
        accept = bool(decision['accept'])
        trace.append(dict(step=t, p_clear=float(decision.get('p_clear',0)),p_same=float(decision.get('p_same',0)),
                          box=box, state=str(decision.get('memory_state','old')), accept=accept))
        if arm in ('memory','full'):
            trace[-1].update(p_initial=decision.get('p_initial'), p_relook=decision.get('p_relook'),
                             contamination_risk=getattr(model,'contamination_risk',None))
        if accept:
            assert box is not None and 0<=box[0]<box[2]<=192 and 0<=box[1]<box[3]<=192
            assert trace[-1]['p_clear']*trace[-1]['p_same']>=.835-1e-9
            submit=t
            break
    destination.mkdir(parents=True,exist_ok=True)
    digest = None
    if submit is not None:
        image=rgb[submit]
        l,u,r,b=box
        Image.fromarray(image).save(destination/'frame.png')
        Image.fromarray(image[u:b,l:r]).save(destination/'crop.png')
        digest=hashlib.sha256(image.tobytes()).hexdigest()
    output=dict(step=submit,actions_used=t,box=None if submit is None else box,
                status='abstained' if submit is None else 'confirmed', frame_sha256=digest, trace=trace)
    partial=destination/'decision.partial'
    partial.write_text(json.dumps(output,allow_nan=False))
    partial.replace(destination/'decision.json')


def run(arm, house_number):
    os.umask(0o077)
    houses=json.loads((HERE.parent/'new_view_20261129'/'view_design.json').read_text())['cohorts']['blind']
    house=houses[house_number]
    paths=sorted((HERE/'private_inputs'/house).glob('*.npz'))
    assert len(paths) in (26,31,37)
    for path in paths:
        dest=HERE/'private_paired'/arm/house/path.stem
        if not (dest/'decision.json').exists():
            apply_arm(path,dest,arm)
    print(arm,house,len(paths),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('arm',choices=ARMS)
    parser.add_argument('house',type=int)
    args=parser.parse_args()
    run(args.arm,args.house)
