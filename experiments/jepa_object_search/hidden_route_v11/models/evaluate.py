"""Freeze all heads. Rank blind candidates before consulting branch opportunity labels."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np
import torch
from train_compare import Ranker, SEEDS, TAPS, memory_start, bonus, remem, route_inputs, sha, write


def clock_memory(mem,row,first,teacher):
    """Reinsert only the branch the policy actually selected in the two-probe panel."""
    memory_start(mem,row['cue'][:,row['q']])
    a=row['plans'][first];xy=np.zeros(2);angle=0.
    taps={round(f*len(a)):i for i,f in enumerate(TAPS)}
    if 'collided' not in teacher:
        return None
    collision=teacher['collided'][first].astype(bool)
    clues=teacher['future_cue'][first,:,row['q']]
    for t,act in enumerate(a,1):
        if act==0 and not collision[t-1]:
            xy+=.25*np.array([math.cos(angle),math.sin(angle)])
        if act:angle+=(1 if act==1 else -1)*math.pi/6
        mem.observe(xy,2+t,[math.cos(angle),math.sin(angle)],float(clues[taps[t]]) if t in taps else None,bool(collision[t-1]))
    # Common-origin two-probe evaluation assumes a return to the branch point.
    # Its outcome cannot imply that a natural-scene return was successfully executed.
    elapsed=40 if row['src']=='blender' and len(a)==28 else len(a)
    return bonus(mem,row['plans'],2+len(a)+elapsed)


@torch.inference_mode()
def evaluate(root,split):
    d=root/'jepa_hidden_v11/models'; rec=json.loads((d/'fit_receipt.json').read_bytes()); models={}
    for mode in rec:
        for info in rec[mode]['runs']:
            arm,seed=info['arm'],info['seed'];p=d/'weights'/mode/f'{arm}_{seed}.pt'
            assert sha(p)==info['checkpoint_sha256']
            ck=torch.load(p,map_location='cpu',weights_only=True)
            m=Ranker(ck['goal.weight'].shape[1]);m.load_state_dict(ck);models[mode,arm,seed]=m.eval()
    mem=remem(root);out=[];torch.set_num_threads(2)
    for src in ('hssd','blender'):
        folder=d/'cache'/src/split
        for p in sorted((folder/'actor').glob('*.npz')):
            with np.load(p) as z:
                acts=z['actions'];ctx=z['ctx'];words=z['words'].tolist();tt=torch.tensor(z['text']);cue=z['cue']
                x=torch.tensor(route_inputs(ctx.astype('float32'),acts))[None].expand(len(words),-1,-1,-1)
                house,origin=str(z['house']),str(z['origin'])
            assert house not in {h[1] for mm in rec.values() for h in mm['houses'] if h[0]==src}
            frozen={k:m(x,tt)[0].sigmoid().numpy() for k,m in models.items()}
            predicted=[]
            for q,word in enumerate(words):
                row=dict(q=q,plans=acts,cue=cue,src=src)
                memory_start(mem,cue[:,q])
                m0=bonus(mem,acts,2); after={}
                for mode,arm,seed in frozen:
                    sc=frozen[mode,arm,seed][q];first=int(np.argmax(sc+m0))
                    if first not in after:
                        with np.load(folder/'teacher'/p.name) as seen:
                            after[first]=clock_memory(mem,row,first,seen)
                    m1=after[first];second=None
                    if m1 is not None:
                        value=sc+m1;value[first]=-np.inf;second=int(np.argmax(value))
                    predicted.append(dict(source=src,split=split,house=house,origin=origin,q=q,query=word,
                        mode=mode,arm=arm,seed=seed,score=(sc+m0).tolist(),memory_bonus=m0.tolist(),
                        first=first,second=second,first_without_memory=int(np.argmax(sc)),horizon=acts.shape[1]))
            # Both arms and all seeds have chosen before any candidate ground truth is loaded.
            with np.load(folder/'truth'/p.name) as truth:
                for row in predicted:
                    q=row.pop('q');y=truth['y'][:,q]
                    row.update(gain=y.tolist(),group=str(truth['group'][q]),candidates=len(y))
                    if 'room_instance_counts' in truth:
                        counts=truth['room_instance_counts'];row['room_instance_count']=int(counts[q]) if counts.ndim else int(counts)
            out.extend(predicted)
    path=d/(split+'_predictions.jsonl')
    with path.open('w') as f:
        for row in out:f.write(json.dumps(row,allow_nan=False)+'\n')
    print(split,len(out),flush=True)


def stats(d,split):
    data=[json.loads(l) for l in (d/(split+'_predictions.jsonl')).read_text().splitlines()]
    records=defaultdict(dict)
    for row in data:
        k=(row['source'],row['house'],row['origin'],row['query'],row['mode'],row['seed'])
        if row['arm'] in records[k]:raise ValueError(f'duplicate {k}')
        records[k][row['arm']]=row
    groups=defaultdict(list)
    for key,pair in records.items():
        j,ds=pair['jepa'],pair['direct'];assert j['gain']==ds['gain'] and j['group']==ds['group']
        y=np.asarray(j['gain']);score=[]
        for r in (j,ds):
            f,s=r['first'],r['second']
            score.append((float(y[f]>0),float(y[f]),None if s is None else float(max(y[f],y[s])>0)))
        meta=dict(house=key[1],origin=key[2],seed=key[-1],score=score,changed=j['first']!=ds['first'],y=y.tolist())
        tags=['all_positive'] if y.max()>0 else ['absent_or_zero_opportunity']
        tags+=['registered:'+str(j['group'])]
        if np.ptp(y)>0:tags+=['informative']
        if np.sum(y>0)>=2:tags+=['multiple_profitable_routes']
        if 'room_instance_count' in j and j['room_instance_count']>1:tags+=['multiple_instances_in_house']
        for tag in tags:groups[key[0],key[-2],tag].append(meta)
    summary={}
    for (src,mode,g),rows in groups.items():
        houses=sorted({r['house'] for r in rows});z={};delta=[]
        for h in houses:
            v=[r for r in rows if r['house']==h]
            av=np.array([[s[0] for s in r['score']] for r in v]).mean(axis=0)
            z[h]=av.tolist();delta.append(av[0]-av[1])
        avg=np.array(list(z.values())).mean(axis=0)
        rg=np.random.default_rng(971)
        boot=rg.choice(delta,(10000,len(delta)),replace=True).mean(1)
        def overall(i):
            return np.asarray([[r['score'][a][i] for a in (0,1)] for r in rows],dtype=float).mean(0).tolist()
        v=overall(2)
        summary['|'.join((src,mode,g))]=dict(origins=len({(r['house'],r['origin']) for r in rows}),
            query_origins=len(rows)//len(SEEDS),houses=len(houses),seeds=list(SEEDS),
            first_success_house_macro_jepa_direct=avg.tolist(),first_success_micro_jepa_direct=overall(0),
            first_gain_micro_jepa_direct=overall(1),two_probe_success_micro_jepa_direct=v if np.isfinite(v).all() else None,
            house_paired_delta=float(np.mean(delta)),cluster_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),
            discordant_choices=sum(r['changed'] for r in rows),random_uniform_first_success=float(np.mean([np.mean(np.array(r['y'])>0) for r in rows])),
            per_house_macro_jepa_direct=z)
    write(d/(split+'_results.json'),summary)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('project_root',type=Path);ap.add_argument('split',choices=['calibration','test'])
    args=ap.parse_args();evaluate(args.project_root,args.split);stats(args.project_root/'jepa_hidden_v11/models',args.split)
