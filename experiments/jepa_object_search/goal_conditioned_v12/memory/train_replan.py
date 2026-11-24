"""Fit a small pairwise evidence head and select its influence on calibration houses."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from replan import second_scores


def read(root,split):
    d=root/'memory/panels'
    with np.load(d/(split+'_inputs.npz')) as x,np.load(d/(split+'_outcomes.npz')) as y:
        assert np.array_equal(x['key'],y['key'])
        return x['key'],x['features'],x['bonus'],y['valid'],y['first'],y['union']


def fit(panel):
    keys,feat,_,valid,first_y,union=panel
    diffs, houses = [],[]
    for r,(house,_,_) in enumerate(keys):
        for f,ok in enumerate(valid[r]):
            if not ok:
                continue
            j,k=[b for b in range(3) if b!=f]
            extra=union[r,f]-first_y[r,f]
            if extra[j]!=extra[k]:
                diffs.append(np.sign(extra[j]-extra[k])*(feat[r,f,j]-feat[r,f,k]))
                houses.append(house)
    z=np.stack(diffs)
    count=Counter(houses)
    w=np.array([1./(len(count)*count[h]) for h in houses])
    scale=np.maximum(np.sqrt(w@(z*z)),.02)
    z=z/scale
    def objective(v):
        scores=z@v
        return w@np.logaddexp(0,-scores)+.15*(v@v),z.T@(-w*expit(-scores))+.30*v
    fit=minimize(objective,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B')
    if not fit.success:
        raise RuntimeError(fit.message)
    return dict(scale=scale.tolist(),coef=fit.x.tolist(),fit_houses=len(count),training_pairs=len(w))


def summarize(rows):
    origin=defaultdict(list)
    for r in rows:
        origin[r['house'],r['origin'],r['seed']].append([
            r['first_gain']>0,r['union_gain']>0,r['union_gain']>r['first_gain'],
            r['union_gain'],r['changed'],r['valid']])
    houses=defaultdict(list)
    for (h,_,_), vs in origin.items():
        houses[h].append(np.mean(vs,axis=0))
    hs={h:np.mean(vals,axis=0).tolist() for h,vals in houses.items()}
    overall=np.mean(list(hs.values()),axis=0)
    names=('first_hit','any_hit','new_discovery','instances','changed','valid')
    return dict(houses=len(hs),seeded_cases=len(rows),interrupted=sum(not r['valid'] for r in rows),
                **dict(zip(names,overall.tolist())),by_house=hs)


def score(panel,choices,head,gate):
    keys,feat,bonus,valid,first_y,union=panel
    idx={tuple(k):i for i,k in enumerate(keys)}
    rows=[]
    for r in choices:
        h,o,q=r['house'],r['origin'],r['query']
        i=idx[h,o,q]; f=r['first']
        initial=np.asarray(r['score'],float); m=np.asarray(r['initial_memory'],float)
        assert int(np.argmax(initial))==f
        chosen=None;changed=False
        if valid[i,f]:
            scores=second_scores(initial,m,bonus[i,f],feat[i,f],dict(head,gate=gate),f)
            chosen=int(np.argmax(scores))
            base=initial-m+bonus[i,f];base[f]=-np.inf
            changed=chosen!=int(np.argmax(base))
        value=int(first_y[i,f] if chosen is None else union[i,f,chosen])
        assert value>=first_y[i,f]
        rows.append(dict(house=h,origin=o,query=q,seed=r['seed'],first=f,second=chosen,
                         first_gain=int(first_y[i,f]),union_gain=value,valid=chosen is not None,changed=changed))
    if len({(r['house'],r['origin'],r['query'],r['seed']) for r in rows})!=3*len(keys):
        raise ValueError('incomplete seeded coverage')
    return summarize(rows)


def load_choices(path):
    table=defaultdict(list)
    for l in path.read_text().splitlines():
        row=json.loads(l)
        table[row['arm']].append(row)
    return table


def main(a):
    root=a.experiment;d=root/'memory'
    if a.command=='fit':
        head=fit(read(root,'train'))
        panel=read(root,'calibration')
        choices=load_choices(a.choices)[a.arm]
        history=[]
        for gate in (0.,.25,.5,1.):
            m=score(panel,choices,head,gate)
            history.append(dict(gate=gate,objective=.65*m['new_discovery']+.35*m['any_hit'],
                                new_discovery=m['new_discovery'],any_hit=m['any_hit']))
        # Ties select the smaller perturbation.
        head['gate']=max(history,key=lambda r:r['objective'])['gate']
        dst=d/'trained_replan.json'
        record=dict(head,calibration=history,selection_arm=a.arm)
        if dst.exists() and json.loads(dst.read_text())!=record:
            raise RuntimeError('frozen memory head already present')
        dst.write_text(json.dumps(record,indent=2))
        print('fit',head['training_pairs'],head['gate'],flush=True)
        return
    head=json.loads((d/'trained_replan.json').read_text());panel=read(root,a.split);summary={}
    for arm,choices in load_choices(a.choices).items():
        summary[arm]={'shared_online_memory':score(panel,choices,head,0.),
                      'trained_update':score(panel,choices,head,head['gate'])}
    (d/a.output).write_text(json.dumps(summary,indent=2))
    print(list(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('experiment',type=Path);p.add_argument('command',choices=('fit','eval'))
    p.add_argument('choices',type=Path);p.add_argument('--arm',default='v12_jepa')
    p.add_argument('--output',default='fresh_memory_summary.json')
    p.add_argument('--split',choices=('fresh','fresh2'),default='fresh')
    main(p.parse_args())
