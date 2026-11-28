"""Counterfactual nonstop routes matched at local request, scene, initial actions and budget."""
import collections,json,sys,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'new_view_20261129'
LEGACY=HERE.parent/'reactive_view_20261129/private_roll'
V1=HERE.parent/'layout_detour_20261129/private_motion'
def paired(old,new):
    return dict(n=len(old),before=sum(old),after=sum(new),gained=sum(not a and b for a,b in zip(old,new)),lost=sum(a and not b for a,b in zip(old,new)))
def main():
    clear=[[],[],[]]; knowns=[];houses=[];data={}
    stats=[collections.Counter() for _ in clear]
    for f in sorted((HERE/'private_motion').glob('*/*.json')):
        rel=f.relative_to(HERE/'private_motion')
        arms=[json.loads(g.read_text()) for g in (LEGACY/rel,V1/rel,f)]
        b=arms[0]['budget'];prefix=arms[0]['baseline_length']
        house=arms[0]['house'];houses.append(house)
        forms=[]
        for i,r in enumerate(arms):
            assert r['group']==arms[0]['group'] and r['focus'] and r['budget']==b
            events=r['events'];assert len(events)-1<=b
            for a,c in zip(arms[0]['events'][:prefix+1],events[:prefix+1]):
                assert a['action']==c['action'] and abs(a['px']-c['px'])<=1
            recovery=events[prefix+1:]
            clear[i].append(any(e['px']>=369 for e in events))
            counts=stats[i]
            counts['actions']+=len(events)-1;counts['recovery_forward_attempts']+=sum(e['action']==1 for e in recovery)
            counts['recovery_stalls']+=sum(e['bump'] for e in recovery)
            counts['recovery_groups_with_stall']+=any(e['bump'] for e in recovery)
            counts['turns_after_initial']+=sum(e['action'] in (2,3) for e in recovery)
            counts['survey_decisions']+=sum(q['details']['reason']=='survey_unseen_side' for q in r['replans'])
            if i==2:
                counts['groups_using_known_grid']+=any(q['details']['reason']=='observed_grid_waypoint' for q in r['replans'])
                # A waypoint choice can run in several segments with a budgeted rescan between them.
                curved=False;headings=set();chosen_curves=0;executed_curves=0
                for k,q in enumerate(r['replans']):
                    details=q['details']
                    if details['reason']!='resume_grid_waypoint':
                        executed_curves+=int(curved and len(headings)>=2)
                        headings=set()
                        curved=details.get('mode')=='curved'
                        chosen_curves+=int(curved)
                    end=r['replans'][k+1]['t'] if k+1<len(r['replans']) else len(events)-1
                    interval=events[q['t']+1:end+1]
                    assert [e['action'] for e in interval]==q['actions'][:len(interval)]
                    for e in interval:
                        if e['action']==1 and not e['bump']:headings.add(round(e['pose'][2]*6/math.pi)%12)
                executed_curves+=int(curved and len(headings)>=2)
                counts['curved_waypoints_selected']+=chosen_curves
                counts['groups_selecting_curved_waypoint']+=bool(chosen_curves)
                counts['curved_waypoints_traversed_at_least_two_bearings']+=executed_curves
                counts['groups_traversing_curved_waypoint']+=bool(executed_curves)
        if house not in data:
            with np.load(BASE/'private_render_blind60'/('blind_'+house)/'controls.npz') as d:
                data[house]={k:d[k] for k in ('group','action','reach','pixels')}
        d=data[house]
        knowns.append(bool(np.any((d['group']==arms[0]['group'])&d['reach'].astype(bool)&(d['pixels']>=369)&((d['action']>0).sum(1)+2<=b))))
    assert len(houses)==94
    def compare(idx,mask):return paired([a for a,m in zip(clear[idx],mask) if m],[a for a,m in zip(clear[2],mask) if m])
    out=dict(legacy_vs_grid=compare(0,[True]*94),curve_v1_vs_grid=compare(1,[True]*94),known_feasible_legacy_vs_grid=compare(0,knowns),
             by_house=[compare(0,[h==house for h in houses]) for house in dict.fromkeys(houses)],
             legacy=stats[0],curves_v1=stats[1],observed_grid=stats[2],
             scope='Nonstop route only. Reaching sufficient geometric coverage is independent of actual report. All primitive scans charged. Stalls measured from unchanged initial phase onward.')
    (HERE/'route_geometry.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
