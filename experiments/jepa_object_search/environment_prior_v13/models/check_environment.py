"""Test future-room forecasts and target-dependent gains on calibration houses."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import numpy as np

from env_routes import EnvironmentRoutes


def run(project):
    r=project/'jepa_env_v13'
    cache=project/'jepa_hidden_v11/models/cache/hssd/calibration'
    metrics=defaultdict(lambda:defaultdict(list));matched=0;real_examples=[]
    room_acc=defaultdict(lambda:defaultdict(list))
    for seed in (11,29,47):
        api=EnvironmentRoutes(project,seed)
        for p in sorted((cache/'actor').glob('*.npz')):
            with np.load(p) as a:
                h,o=str(a['house']),str(a['origin'])
                plans=a['actions'];ctx=a['ctx'];text=a['text'];words=a['words'].tolist();cue=a['cue']
            with np.load(r/'data/labels/calibration'/(o+'.npz')) as f:
                future=f['view_room'].mean(1)
                mass=future.sum(-1)
                observed=mass>1e-7
                future=future/np.maximum(mass[:,None],1e-7)
            with np.load(cache/'truth'/p.name) as f:
                gain=f['y']
            variable=len(set(future[observed].argmax(-1)))>1
            for q,word in enumerate(words):
                result=api.cached(ctx,text[q],plans,word)
                if q==0:
                    for arm in ('jepa','direct','copy'):
                        dist=result['env_pred_'+arm]
                        if np.any(observed):
                            room_acc[arm][h].append(np.stack([(dist.argmax(-1)==future.argmax(-1)).astype(float),
                                             -(future*np.log(np.maximum(dist,1e-8))).sum(-1)],-1)[observed].mean(0))
                ver=api.old.remem(project);api.old.memory_start(ver,cue[:,q]);bonus=api.old.bonus(ver,plans,2)
                for arm in ('jepa','direct','v11','semantic_jepa','semantic_direct'):
                    # For the unblended semantic diagnostic use expected inventory only.
                    choice=np.argmax(result[arm]+(0 if arm.startswith('semantic_') else bonus))
                    for g in ['all','changing' if variable else 'same']:
                        metrics[(arm,g)][h,o].append(float(gain[choice,q]>0))
                if seed==11 and h not in real_examples:
                    raw_dir=project/'jepa_hidden_v11/hssd/calibration'/h
                    with np.load(raw_dir/(o+'.npz')) as f:
                        rgb=f['ctx']
                    live=api.live(rgb,word,plans)
                    if max(np.max(abs(result[k]-live[k])) for k in ('jepa','direct','v11','env_pred_jepa'))>2e-4:
                        raise ValueError('fresh RGB and encoded inputs disagree')
                    prob,seen=api.observe_batch(rgb,word)
                    assert prob.shape==(3,len(api.rooms)) and np.allclose(live['cue'],seen)
                    real_examples.append(h);matched+=1
    out=dict(fresh_rgb_replay_houses=matched,first={},rooms={})
    for (arm,g),values in metrics.items():
        houses=defaultdict(list)
        for (h,_),vs in values.items():houses[h].append(np.mean(vs))
        out['first'][arm+'|'+g]=dict(houses=len(houses),origins=len(values),
            hits=float(np.mean([np.mean(v) for v in houses.values()])))
    for arm,vals in room_acc.items():
        v=np.mean([np.mean(rows,0) for rows in vals.values()],0)
        out['rooms'][arm]=dict(route_dominant_room_accuracy=float(v[0]),visible_distribution_cross_entropy=float(v[1]))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('project',type=Path)
    a=p.parse_args();path=a.project/'jepa_env_v13/models/diagnostics.json'
    path.write_text(json.dumps(run(a.project),indent=2))
