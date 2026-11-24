"""Select a second route using the actually inspected first route and its return observations."""
from collections import defaultdict
import argparse
import json
import math
from pathlib import Path
import numpy as np
from train_compare import memory_start, bonus, remem, write
from encode import Encoder


def advance(mem, xy, yaw, t, actions, blocked, clue):
    for j,a in enumerate(actions,1):
        if a==0 and not blocked[j-1]:
            xy=xy+.25*np.array([math.cos(yaw), math.sin(yaw)])
        if a==1:yaw+=math.pi/6
        if a==2:yaw-=math.pi/6
        if a not in (0,1,2,3,-1):raise ValueError(f'unexpected return command {a}')
        t+=1
        mem.observe(xy,t,[math.cos(yaw),math.sin(yaw)],clue.get(j),bool(blocked[j-1]))
    return xy,yaw,t


def run(root):
    exp=root/'jepa_hidden_v11';d=exp/'models'; validation=exp/'validation'
    records=defaultdict(list)
    for l in (d/'test_predictions.jsonl').read_text().splitlines():
        r=json.loads(l)
        if r['source']=='hssd':records[r['house'],r['origin']].append(r)
    mem=remem(root); enc=Encoder(root);out=[]
    loc=d/'cache/hssd/test'
    for path in sorted((loc/'actor').glob('*.npz')):
        with np.load(path) as actor:
            key=str(actor['house']),str(actor['origin'])
            rows=records[key];assert rows
            acts,words,cue=actor['actions'],actor['words'].tolist(),actor['cue']
        rt=validation/'returns'/key[0]/(key[1]+'.npz')
        gt=validation/'returns_truth'/key[0]/(key[1]+'.npz')
        with np.load(loc/'teacher'/path.name) as seen:
            forward_rb,forward_v=seen['collided'],seen['future_cue']
        chosen=[]
        with np.load(rt) as f:
            reading={}
            for row in rows:
                first=row['first'];q=words.index(row['query'])
                valid=bool(f['return_valid'][first])
                second=None
                if valid:
                    if first not in reading:
                        rgb=f['return_rgb'][first]
                        _,visible=enc.encode(rgb,enc.words(words))
                        reading[first]=visible
                    memory_start(mem,cue[:,q])
                    n=len(acts[first]);taps=np.rint(np.arange(1,5)*n/4).astype(int)
                    xy,yaw,t=advance(mem,np.zeros(2),0.,2,acts[first],forward_rb[first],dict(zip(taps,forward_v[first,:,q])))
                    back=f['return_actions'][first];block=f['return_collided'][first];v=reading[first][:,q]
                    step_ids=np.rint(np.arange(1,len(v)+1)*len(back)/len(v)).astype(int)
                    xy,yaw,t=advance(mem,xy,yaw,t,back,block,dict(zip(step_ids,v)))
                    # Align the original bearings with the estimated return frame. Matched
                    # physical return validity is independently checked by the simulator.
                    m=mem.features(acts,xy,[math.cos(yaw),math.sin(yaw)],t)
                    term=m@np.array([mem.weights[k] for k in ('novelty','clearance','target_trace')])
                    rank=np.array(row['score'])-row['memory_bonus']+term
                    rank[first]=-np.inf;second=int(np.argmax(rank))
                chosen.append((row,q,first,second))
        # Physical outcome grid is read only after both decisions have been committed.
        with np.load(gt) as truth:
            assert truth['words'].tolist()==words
            for row,q,first,second in chosen:
                first_gain=int(truth['first_y'][first,q]);assert first_gain==row['gain'][first]
                union=first_gain if second is None else int(truth['union_y'][first,second,q])
                out.append(dict(house=key[0],origin=key[1],query=words[q],mode=row['mode'],arm=row['arm'],
                    seed=row['seed'],first=first,second=second,return_valid=second is not None,
                    first_gain=first_gain,union_gain=union))
    with (d/'physical_returns.jsonl').open('w') as f:
        for row in out:f.write(json.dumps(row)+'\n')
    result={}
    for mode in ('hssd','combined'):
        for arm in ('jepa','direct','wrong_future'):
            select=[r for r in out if (r['mode'],r['arm'])==(mode,arm)]
            groups=defaultdict(list)
            for r in select:groups[r['house']].append(r)
            result[mode+'|'+arm]=dict(rows_including_three_seeds=len(select),
                invalid_return=sum(not r['return_valid'] for r in select),
                hit_rate_micro=float(np.mean([r['union_gain']>0 for r in select])),
                house_macro=float(np.mean([np.mean([r['union_gain']>0 for r in v]) for v in groups.values()])),
                mean_unique_instances=float(np.mean([r['union_gain'] for r in select])))
        for ref in ('direct','wrong_future'):
            vals=[]
            for h in sorted({r['house'] for r in out}):
                picks=[[r['union_gain']>0 for r in out if (r['mode'],r['arm'],r['house'])==(mode,a,h)] for a in ('jepa',ref)]
                if len(picks[0])!=len(picks[1]):raise ValueError('unpaired house')
                vals.append(np.mean(picks[0])-np.mean(picks[1]))
            boot=np.random.default_rng(971).choice(vals,(10000,len(vals)),replace=True).mean(1)
            result[mode+'|jepa_minus_'+ref]=dict(houses=len(vals),paired_house_mean=float(np.mean(vals)),
                    house_bootstrap_95=np.quantile(boot,[.025,.975]).tolist())
    write(d/'physical_results.json',result)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('project_root',type=Path);run(ap.parse_args().project_root)
