"""Replay one fixed calibration decision per house through the public RGB update API."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from live_update import ObservedRouteUpdate
from replan import second_scores


def run(project):
    old=project/'jepa_hidden_v11'; new=project/'jepa_route_v12'
    sys.path.insert(0,str(old/'models'))
    from train_compare import remem,memory_start
    api=ObservedRouteUpdate(project)
    choices={}
    for line in (new/'models/calibration_choices.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['arm']=='v12_jepa' and r['seed']==11:
            choices[r['house'],r['origin'],r['query']]=r
    result={}
    with np.load(new/'memory/panels/calibration_inputs.npz') as panel:
        for i,key in sorted(enumerate(panel['key'].tolist()),key=lambda p:p[1]):
            h,o,q=key
            if h in result:
                continue
            row=choices[h,o,q]; f=row['first']; m=np.array(row['initial_memory'])
            expected=second_scores(np.array(row['score']),m,panel['bonus'][i,f],panel['features'][i,f],api.head,f)
            with np.load(old/'hssd/calibration'/h/(o+'.npz')) as raw, np.load(
                     new/'continuation/calibration/validation/returns'/h/(o+'.npz')) as home:
                rgb=raw['ctx']; tok=api.encoder.words([q])
                _,cue=api.encoder.encode(rgb,tok)
                memory=remem(project);memory_start(memory,cue[:,0])
                valid=bool(home['return_valid'][f])
                action=api.select(memory=memory,target=q,start_rgb=rgb,plans=raw['actions'],first=f,
                    scores_at_start=row['score'],bonus_at_start=m,outbound_rgb=raw['future'][f],
                    outbound_collisions=raw['collided'][f],return_actions=home['return_actions'][f],
                    return_collisions=home['return_collided'][f],return_rgb=home['return_rgb'][f],
                    start_xy=[0.,0.],start_yaw=0.,start_time=2,returned=valid)
                if action!=(int(np.argmax(expected)) if valid else None):
                    raise AssertionError('live and cached choices differ')
                result[h]=valid
    return dict(calibration_houses=len(result),matching_choices_or_valid_interruptions=len(result),
                valid_returns=sum(result.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('project',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.write_text(json.dumps(run(a.project),indent=2))
