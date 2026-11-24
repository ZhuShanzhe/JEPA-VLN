"""Relate actual discoveries to opportunities, extra steps and training-room specificity."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from analyze import macro


def run(exp):
    sys.path.insert(0,str(exp/'simulator'))
    from oracle import one
    d=exp/'online'
    rows=[json.loads(s) for s in (d/'fresh_horizon_decisions.jsonl').read_text().splitlines()]
    odds=json.loads((exp/'data/prior.json').read_text())
    prob=odds['p_target_given_room']; freq=np.asarray(odds['p_room_prior'])
    limits={}
    for f in sorted((exp/'simulator/trees/fresh').glob('*/*/episode.json')):
        meta=json.loads(f.read_text())
        for (w,b),res in one(meta,json.loads((f.parent/'truth.json').read_text())).items():
            limits[meta['house'],meta['origin'],w,b]=res['best']
    result={}
    for name in ('v11_baseline','jepa_full','jepa_horizon','direct_horizon'):
        rr=[r for r in rows if r['method']==name and r['quota']>0]
        def short(data):
            e=macro(data)['present']
            return dict(houses=e['houses'],queries=e['queries'],any112=e['values']['any'][-1],
                        distinct112=e['values']['distinct'][-1])
        profiles=defaultdict(list)
        for r in rr:
            pr=np.asarray(prob.get(r['word'],freq*0))
            profiles['specific' if (pr.max()-pr@freq)>=.30 else 'diffuse'].append(r)
        counts=[]
        for b,i in [(28,0),(56,1),(84,2),(112,3)]:
            ok=[r for r in rr if limits[r['house'],r['origin'],r['word'],b]>=1]
            assert all(r['found'][i]<=limits[r['house'],r['origin'],r['word'],b] for r in rr)
            counts.append({'budget':b,'reachable_queries':len(ok)//3,'reachable_seeded_runs':len(ok),
                           'confirmed_seeded_runs':sum(r['found'][i]>0 for r in ok),
                           'micro_fraction_of_attainable_first':sum(r['found'][i]>0 for r in ok)/max(1,len(ok))})
        result[name]=dict(opportunity=counts,by_prior={g:short(v) for g,v in profiles.items()})
    by_key=defaultdict(dict)
    for row in rows:
        if row['quota'] and row['method'] in ('jepa_horizon','v11_baseline','direct_horizon'):
            by_key[row['house'],row['origin'],row['word'],row['seed']][row['method']]=row
    for b in ('v11_baseline','direct_horizon'):
        paired=[(v['jepa_horizon'],v[b]) for v in by_key.values()]
        g=np.array([x['found'][-1]-y['found'][-1] for x,y in paired])
        result['jepa_horizon_minus_'+b]=dict(seed_cases=len(g),more=int(sum(g>0)),same=int(sum(g==0)),
                                               fewer=int(sum(g<0)))
    return result


if __name__=='__main__':
    root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.write_text(json.dumps(run(root),indent=2))
