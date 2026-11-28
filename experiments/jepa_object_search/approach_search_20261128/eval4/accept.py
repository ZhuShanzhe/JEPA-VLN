"""Validate frozen inputs and report per-house paired approach outcomes."""
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
P=HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    panel=json.loads((HERE/'panel.json').read_text())
    prior=json.loads((P/'prior_approach_20261128/learned_prior.json').read_text())
    cohorts=json.loads((P/'jepa_env_v13/simulator/cohort.json').read_text())
    houses={r['scene'] for r in panel}
    assert len(houses)==len(panel)==8
    assert houses.isdisjoint({r['scene'] for r in json.loads((HERE.parent/'eval/panel.json').read_text())})
    assert houses.isdisjoint({r['scene'] for r in json.loads((HERE.parent/'eval2/panel.json').read_text())})
    assert houses.isdisjoint(prior['train_houses'])
    assert houses.isdisjoint(cohorts['calibration']) and houses.isdisjoint(cohorts['fresh'])
    rows=[]
    paths={}
    for i,ep in enumerate(panel):
        trio=[]
        for arm in ['old','new','prior_only']:
            path=HERE/'runs'/(str(i)+'_'+arm+'.json')
            r=json.loads(path.read_text())
            assert (r['scene'],r['episode'],r['goal'],r['steps'],r['arm'])==(ep['scene'],ep['id'],ep['goal'],ep['budget'],arm)
            assert r['quota_met']==(r['distinct_approached']>=ep['quota'])
            assert r['reachable_instances_at_start'] >= 2
            assert r['distinct_approached']==len(r['first_steps'])<=r['instances_available']
            assert all(0<=s<=300 for s in r['first_steps'])
            paths[path.name]=digest(path)
            trio.append(r)
        assert len({r['already_at_start'] for r in trio})==len({r['instances_available'] for r in trio})==1
        rows+=trio
    summaries={}
    for arm in ('old','new','prior_only'):
        group=[r for r in rows if r['arm']==arm]
        summaries[arm] = dict(total_distinct=sum(r['distinct_approached'] for r in group),
            after_departure=sum(r['distinct_approached']-r['already_at_start'] for r in group),
            enough=sum(r['quota_met'] for r in group), visited_at_least_one=sum(r['distinct_approached']>0 for r in group),
            choices=sum(r['stats'].get('chosen',0) for r in group), destinations_reached=sum(r['stats'].get('arrived',0) for r in group),
            collisions=sum(r['stats'].get('collisions',0) for r in group), physical_steps=sum(r['steps'] for r in group),
            stages={k:sum(r['stats'].get(k,0) for r in group) for k in ('prior_search','remaining_search')})
    count=lambda arm:[r['distinct_approached'] for r in rows if r['arm']==arm]
    d=[v-w for v,w in zip(count('new'),count('old'))]
    freeze=json.loads((HERE/'locked_sources.json').read_text())
    for f,val in freeze.items():
        if digest(P/f)!=val:raise RuntimeError(f'Changed after freeze: {f}')
    out=dict(protocol='HSSDOV-derived multi-instance approach pilot, objective quota two; 300 physical actions per arm, oracle used only for trajectory scoring',
        houses=len(houses),prior_train_overlap=0,encoder_calibration_and_fresh_overlap=0,
        runs=len(rows),arms=summaries,per_house_new_minus_old=d,
        paired_better_equal_worse=[sum(x>0 for x in d),sum(x==0 for x in d),sum(x<0 for x in d)],
        result_hashes=paths,code_and_frozen_input_hashes=freeze,rows=rows)
    (HERE/'accepted.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(summaries))

if __name__=="__main__":main()
