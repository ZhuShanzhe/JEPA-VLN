"""Reconcile paired physical runs with separate trace-level diagnostics and no model access."""
import argparse
from pathlib import Path
from collections import Counter
import json
import hashlib


def read(path):
    return json.loads(path.read_text())


def check_public(root):
    panel=read(root/'panel.json')
    result=read(root/'paired_public.json')
    rows=result['by_house']
    assert len(rows)==len(panel)==8 and len({r['scene'].split('_')[0] for r in panel})==8
    assert len(rows)*2==result['accepted_arms']
    for r,s in zip(rows,panel):
        assert (r['scene'],r['goal'])==(s['scene'],s['goal']) and s['budget']==300
    for arm in ('previous','local'):
        s=result['allocation'][arm]
        for field,result_key in [('distinct_approached','instances_at_025m'),('approached_half_m','instances_at_05m'),
                                 ('approached_one_m','instances_at_1m'),('quota_met','houses_at_least_two_at_025m')]:
            assert sum(r[arm][field] for r in rows)==s[result_key]
        assert sum(r[arm]['stats']['steps'] for r in rows)==s['actions']
        assert sum(r[arm]['stats'].get('collisions',0) for r in rows)==s['collisions']
    print('Verified eight pairs, frozen house list and published aggregate metrics.')


def event_metrics(events):
    decisions=[e for e in events if 'action_source' in e]
    if decisions:
        first=next(i for i,e in enumerate(decisions) if e['step']==4)
        assert first==0
        assert all(b['step']>=a.get('last_step',a['step']) for a,b in zip(decisions,decisions[1:]))
    byid={}
    for e in decisions:
        if e.get('action_source')=='local':
            byid.setdefault(e['goal_id'],[]).append(e)
    multi=[es for es in byid.values() if len(es)>=2]
    for es in byid.values():
        site=es[0]['site_xy']
        assert all(e['site_xy']==site for e in es),'target moved between decisions'
        assert all(e['age']==i+1 for i,e in enumerate(es)),'target age discontinuity'
    # Do not count a non-scanning continuation as a new visual evidence query.
    eligible=[e for e in decisions if (e['region_entered'] or e['action_source']=='local') and e['region_phase']!='retained_goal_geometry_check']
    return dict(decisions=len(decisions), independent_targets=len(byid), held_at_least_twice=len(multi),
            held_at_least_three=sum(len(es)>=3 for es in byid.values()),
            maximum_decisions=max((len(es) for es in byid.values()),default=0),
            eligible_image_decisions=len(eligible),images_with_patch=sum(e['patches']>0 for e in eligible),
            images_with_projected_anchor=sum(e['anchors']>0 for e in eligible),
            images_with_proposed_site=sum(e['sites']>0 for e in eligible),
            geometry_only_decisions=sum(e['region_phase']=='retained_goal_geometry_check' for e in decisions),
            physical_actions=round(sum(e.get('last_step',e['step'])-e['step'] for es in byid.values() for e in es)),
            nonzero_local_progress=sum(len(es)>=2 and any(e['distance_m']+.1 < es[0]['distance_m'] for e in es[1:]) for es in byid.values()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--public-only',action='store_true',help='Check only the public table and summary')
    args=parser.parse_args()
    root=args.root.resolve()
    if args.public_only:
        return check_public(root)
    panel=read(root/'panel.json')
    assert read(root/'family_exposure.json')['all_selected_have_disjoint_numeric_prefix']
    for manifest in ('panel_lock.json','local_lock.json'):
        for name,digest in read(root/manifest).items():
            assert hashlib.sha256((root.parents[1]/name).read_bytes()).hexdigest()==digest,name
    accepted=read(root/'eligibility.json')
    rows=[]
    for i,s in enumerate(panel):
        a,b=[read(root/'runs'/f'{i}_{arm}.json') for arm in ('new','local')]
        check=next(e for e in accepted if (e['id'],e['scene'],e['goal'])==(s['id'],s['scene'],s['goal']))
        for r in (a,b):
            assert r['scene']==s['scene'] and r['episode']==s['id'] and r['goal']==s['goal']
            assert r['steps']==s['budget']==300 and r['reachable_instances_at_start']==check['reachable']>=2
            assert r['already_at_start']==check['initially_close'] and r['distinct_approached']==len(r['first_steps'])
            assert r['quota_met']==(r['distinct_approached']>=2)
            assert r['distinct_approached']<=r['approached_half_m']<=r['approached_one_m']<=r['instances_available']
        ev=[json.loads(line) for line in (root/'private_events'/f'{i}.jsonl').read_text().splitlines()]
        detailed=event_metrics(ev)
        assert detailed['decisions']==b['stats']['decisions']
        assert detailed['independent_targets']==b['stats'].get('acquired',0)
        assert detailed['physical_actions']==b['stats'].get('local_steps',0)
        assert detailed['eligible_image_decisions']==b['stats'].get('eligible_frames',0)
        assert detailed['images_with_projected_anchor']==b['stats'].get('with_anchors',0)
        keys=('distinct_approached','already_at_start','first_steps','quota_met','approached_half_m','approached_one_m','stats')
        rows.append(dict(i=i,scene=s['scene'],goal=s['goal'],reachable=check['reachable'],available=b['instances_available'],
            previous={k:a[k] for k in keys},local={k:b[k] for k in keys},trace=detailed,counts=b['counts'],by_phase=b['distance_by_phase']))
    armstats={}
    for arm in ('previous','local'):
        armstats[arm]=dict(instances_at_025m=sum(r[arm]['distinct_approached'] for r in rows),
                           already_at_start=sum(r[arm]['already_at_start'] for r in rows),
                           houses_at_least_one_at_025m=sum(r[arm]['distinct_approached']>0 for r in rows),
                           houses_at_least_two_at_025m=sum(r[arm]['quota_met'] for r in rows),
                           instances_at_05m=sum(r[arm]['approached_half_m'] for r in rows),
                           houses_at_least_two_at_05m=sum(r[arm]['approached_half_m']>=2 for r in rows),
                           instances_at_1m=sum(r[arm]['approached_one_m'] for r in rows),
                           houses_at_least_two_at_1m=sum(r[arm]['approached_one_m']>=2 for r in rows),
                           actions=300*len(rows),collisions=sum(r[arm]['stats'].get('collisions',0) for r in rows))
    gains=[r['local']['distinct_approached']-r['previous']['distinct_approached'] for r in rows]
    previous_decisions=sum(r['previous']['stats'].get('chosen',0)+r['previous']['stats'].get('empty_decisions',0) for r in rows)
    budgets={'previous':dict(decisions=previous_decisions,scan_actions_inferred_from_locked_loop=4*previous_decisions),
             'local':dict(decisions=sum(r['local']['stats']['decisions'] for r in rows))}
    for key in ('scan_steps','region_steps','local_steps','empty_steps','tail_steps','rescan_saved'):
        budgets['local'][key]=sum(r['local']['stats'].get(key,0) for r in rows)
    assert sum(budgets['local'][k] for k in ('scan_steps','region_steps','local_steps','empty_steps','tail_steps'))==armstats['local']['actions']
    sums=lambda sub,key:sum(r[sub].get(key,0) for r in rows)
    traces={k:sums('trace',k) for k in rows[0]['trace'] if k!='maximum_decisions'}
    traces['maximum_decisions']=max(r['trace']['maximum_decisions'] for r in rows)
    counts={name:dict(sum((Counter(r['counts'].get(name,{})) for r in rows),Counter())) for name in rows[0]['counts']}
    phases={name:{limit:sum(r['by_phase'][name][limit] for r in rows) for limit in rows[0]['by_phase'][name]}
            for name in rows[0]['by_phase']}
    payload=dict(accepted_arms=len(rows)*2,unrelated_october_fgprompt_scenes=len(read(root/'family_exposure.json')['legacy_unrelated_exposure']),
            allocation=armstats,by_house_change=dict(improved=sum(x>0 for x in gains),tied=gains.count(0),declined=sum(x<0 for x in gains)),
            action_budget=budgets,trace_totals=traces,counts=counts,phase_proximity=phases,by_house=rows)
    out=root/'paired_public.json'
    out.write_text(json.dumps(payload,indent=2)+'\n')
    check_public(root)
    print(json.dumps({k:v for k,v in payload.items() if k!='by_house'}))
    for r in rows:
        print(r['goal'],r['scene'],[r[arm][k] for arm in ('previous','local') for k in ('distinct_approached','approached_half_m','approached_one_m')])


if __name__=='__main__':main()
