"""Construct training examples and held-out replay features with separated outcomes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from replan import evidence_features


def run(project, split):
    old, new = project/'jepa_hidden_v11', project/'jepa_route_v12'
    sys.path.insert(0, str(old/'models'))
    from physical_return import advance
    from train_compare import remem, memory_start
    mem = remem(project)
    w = np.asarray([mem.weights[key] for key in ('novelty','clearance','target_trace')])
    source = new if split in ('fresh','fresh2') else old
    disk = source/'models/cache/hssd'/split
    replay = new/'expansion' if split=='fresh2' else new if split=='fresh' else new/'continuation'/split
    features, bonuses, outcomes, indices = [], [], [], []
    for path in sorted((disk/'actor').glob('*.npz')):
        with np.load(path) as a:
            ctx, text = a['ctx'].astype(float),a['text'].astype(float)
            plans,cue,words,house,origin = a['actions'],a['cue'],a['words'].tolist(),str(a['house']),str(a['origin'])
        with np.load(disk/'teacher'/path.name) as e:
            visited_cue,visited,coll = e['future_cue'],e['future'].astype(float),e['collided']
        with np.load(replay/'validation/returns'/house/(origin+'.npz')) as e:
            back, hitwall, valid = e['return_actions'], e['return_collided'], e['return_valid']
        with np.load(new/'memory/cache_return'/split/house/(origin+'.npz')) as e:
            assert e['words'].tolist()==words
            back_cue = e['cue']
        feature_rows, bonus_rows = [], []
        for q,word in enumerate(words):
            ff,bb = [], []
            for first in range(len(plans)):
                memory_start(mem,cue[:,q])
                xy,yaw,t = advance(mem,np.zeros(2),0.,2,plans[first],coll[first],
                                    dict(zip((6,12,18,24),visited_cue[first,:,q])))
                xy,yaw,t = advance(mem,xy,yaw,t,back[first],hitwall[first],
                                    dict(zip((6,12,18,24,30,36),back_cue[first,:,q])))
                evidence = mem.features(plans,xy,[np.cos(yaw),np.sin(yaw)],t)
                ff.append(evidence_features(ctx,text[q],plans,first,visited[first],visited_cue[first,:,q],
                                             cue[:,q],coll[first],evidence))
                bb.append(evidence @ w)
            feature_rows.append(ff)
            bonus_rows.append(bb)
            indices.append([house,origin,word])
        features.extend(feature_rows)
        bonuses.extend(bonus_rows)
        # Outcomes are accessed only after building all policy features for the junction.
        with np.load(replay/'validation/returns_truth'/house/(origin+'.npz')) as e:
            assert e['words'].tolist()==words
            for q in range(len(words)):
                outcomes.append((valid,e['first_y'][:,q], e['union_y'][:,:,q]))
    if not indices or len({tuple(i) for i in indices})!=len(indices):
        raise ValueError('missing or duplicate keys')
    dst=new/'memory/panels';dst.mkdir(exist_ok=True)
    np.savez_compressed(dst/(split+'_inputs.npz'),key=np.asarray(indices), features=features,bonus=bonuses)
    np.savez_compressed(dst/(split+'_outcomes.npz'),key=np.asarray(indices),valid=np.stack([v[0] for v in outcomes]),
                        first=np.stack([v[1] for v in outcomes]),union=np.stack([v[2] for v in outcomes]))
    print(split,len(indices),len({i[0] for i in indices}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('project',type=Path)
    p.add_argument('split',choices=('train','calibration','fresh','fresh2'))
    args=p.parse_args()
    run(args.project,args.split)
