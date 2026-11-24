"""House-paired summaries of confirmed instances at four budgets and calibration selection."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


BUDGETS=(28,56,84,112)
NAMES=('distinct','any','two','quota','first_step_capped','second_step_capped')


def macro(records):
    by={}
    for subset,test in [('present',lambda q:q>0),('multiple',lambda q:q>=2),('single',lambda q:q==1),
                        ('absent',lambda q:q==0)]:
        by_origin=defaultdict(list);identity=set()
        for r in records:
            if not test(r['quota']):continue
            h,o=r['house'],r['origin'];identity.add((h,o,r['word']))
            vals=[]
            for b,n in zip(BUDGETS,r['found']):
                vals.append([n,n>=1,n>=2,r['quota']>0 and n>=r['quota'],
                             min(r['first_step'] or b,b),min(r['second_step'] or b,b)])
            by_origin[h,o].append(vals)
        if not identity:continue
        by_house=defaultdict(list)
        for (h,o),v in by_origin.items():by_house[h].append(np.mean(v,axis=0))
        table={h:np.mean(v,axis=0) for h,v in by_house.items()}
        by[subset]=dict(houses=len(table),origins=len(by_origin),queries=len(identity),
            by_house={h:a.tolist() for h,a in table.items()},
            values={k:v.tolist() for k,v in zip(NAMES,np.mean(list(table.values()),axis=0).T)})
    return by


def table_for(path):
    data=defaultdict(list); seen=set()
    for line in path.read_text().splitlines():
        r=json.loads(line)
        key=(r['method'],r['house'],r['origin'],r['word'],r['seed'])
        if key in seen:raise ValueError('repeated decision trace')
        seen.add(key);data[r['method']].append(r)
    identities={m:{(r['house'],r['origin'],r['word'],r['seed']) for r in rows} for m,rows in data.items()}
    assert len({frozenset(v) for v in identities.values()})==1
    return {m:macro(rows) for m,rows in data.items()},data


def paired(a,b):
    hs=sorted(a['by_house']);assert hs==sorted(b['by_house'])
    d=np.asarray([a['by_house'][h] for h in hs])-np.asarray([b['by_house'][h] for h in hs])
    ind=np.random.default_rng(1313).integers(len(hs),size=(12000,len(hs)))
    boot=d[ind].mean(1)
    out={name:{'delta':d.mean(0)[:,k].tolist(),'ci95':np.quantile(boot[:,:,k],[.025,.975],axis=0).T.tolist()}
         for k,name in enumerate(NAMES)}
    gain=d[:,-1,0]-d[:,0,0]
    out['distinct_extra_gain_112_minus_28']=dict(delta=float(gain.mean()),
                     ci95=np.quantile(gain[ind].mean(1),[.025,.975]).tolist())
    return out


def run(root,split,stage):
    d=root/'online'
    path=d/(split+'_decisions.jsonl' if not stage else split+'_'+stage+'_decisions.jsonl')
    tbl,traces=table_for(path)
    if split=='calibration' and not stage:
        candidates={}
        for m,x in tbl.items():
            v=x['present']['values'];multi=x['multiple']['values']
            candidates[m]=.5*v['distinct'][-1]+.3*multi['two'][-1]+.2*v['any'][1]
        order=['jepa_model_only','jepa_weak','jepa_balanced','jepa_scout','jepa_scout_strong']
        best=max(order,key=lambda m:round(candidates[m],10))
        # Conservative tie preference. Keep explicitly planned scouting as a secondary strategy.
        result=dict(setting=best.removeprefix('jepa_'),formula='0.5 distinct@112 + 0.3 multiple-two@112 + 0.2 present-any@56',
                    candidates=candidates)
        dest=d/'online_selection.json'
        if dest.exists() and json.loads(dest.read_text())!=result:raise RuntimeError('selection is already frozen')
        dest.write_text(json.dumps(result,indent=2))
    pairs={}
    if 'v11_baseline' in tbl:
        compare=[(m,'v11_baseline') for m in tbl if m!='v11_baseline']
        compare += [('jepa_full',m) for m in ('direct_with_update','jepa_prior_only','jepa_without_scene_evidence',
                                                'jepa_model_only') if m in tbl and 'jepa_full' in tbl]
        if 'jepa_horizon' in tbl and 'jepa_full' in tbl:compare.append(('jepa_horizon','jepa_full'))
        if 'direct_horizon' in tbl and 'jepa_horizon' in tbl:compare.append(('jepa_horizon','direct_horizon'))
        for candidate in ('jepa_horizon_layout','jepa_horizon_transition'):
            if candidate in tbl:compare.append((candidate,'jepa_horizon'))
        for m,b in compare:
            pairs[m+'__'+b]={s:paired(tbl[m][s],tbl[b][s]) for s in ('present','multiple')}
    compact={m:{s:{k:v for k,v in row.items() if k!='by_house'} for s,row in a.items()} for m,a in tbl.items()}
    out=dict(budgets=BUDGETS,arms=compact,paired=pairs)
    if 'jepa_dedicated_scout' in traces:
        sel=[r for r in traces['jepa_dedicated_scout'] if r['quota']]
        out['dedicated_inspection_changed_second_choice_micro']=float(np.mean([r['scout_changed'] for r in sel]))
    dest=d/(split+('_'+stage if stage else '')+'_summary.json')
    dest.write_text(json.dumps(out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('experiment',type=Path)
    p.add_argument('split',choices=('calibration','fresh'));p.add_argument('--stage',default='')
    a=p.parse_args();run(a.experiment,a.split,a.stage)
