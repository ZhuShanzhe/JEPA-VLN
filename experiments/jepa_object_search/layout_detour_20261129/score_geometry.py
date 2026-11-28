"""Paired visibility and executed-curvature audit. All instance labels here are evaluator-only."""
import collections,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'reactive_view_20261129'
BASE=HERE.parent/'new_view_20261129'
def delta(a,b):
    return dict(n=len(a),before=sum(a),after=sum(b),
                gained=sum(not x and y for x,y in zip(a,b)),lost=sum(x and not y for x,y in zip(a,b)))
def main():
    arms=[];seen=collections.Counter();house=collections.defaultdict(list)
    for new_file in sorted((HERE/'private_motion').glob('*/*.json')):
        old_file=PRIOR/'private_roll'/new_file.parent.name/new_file.name
        old=json.loads(old_file.read_text());new=json.loads(new_file.read_text())
        assert new['group']==old['group'] and new['focus'] and old['focus']
        budget=new['budget'];assert budget==old['budget']
        phase=new['baseline_length'];geometry=[]
        for old_event,event in zip(old['events'][:phase+1],new['events'][:phase+1]):
            assert old_event['action']==event['action'] and abs(old_event['px']-event['px'])<=1
        for row in (old,new):
            assert len(row['events'])-1 <=budget
            geometry.append(any(e['px']>=369 for e in row['events']))
        selected=executed=False
        for i,replan in enumerate(new['replans']):
            if replan['details']['reason']!='layout_jepa_detour':continue
            selected=True;seen['detour_plans']+=1
            start=replan['t'];end=(new['replans'][i+1]['t'] if i+1<len(new['replans']) else len(new['events'])-1)
            window=new['events'][start+1:end+1];head=set()
            assert [e['action'] for e in window]==replan['actions'][:len(window)]
            for e in window:
                if e['action']==1 and not e['bump']:head.add(round(e['pose'][2]*6/math.pi)%12)
            executed|=len(head)>=2
            seen['physically_bent_plans']+=len(head)>=2
            seen['plans_contacted']+=any(e['bump'] for e in window)
            seen['plans_completed_full_action_sequence']+=len(window)==len(replan['actions'])
        seen['groups_with_layout_selected']+=selected;seen['groups_actually_traversed_bend']+=executed
        old_clear,new_clear=geometry
        if not old_clear and new_clear:
            seen['newly_clear_groups_traversed_bend']+=executed
        house[new['house']].append(geometry)
        # Proof of an independently measured clear action sequence with matching original budget.
        orig=BASE/'private_render_blind60'/('blind_'+new['house'])/'controls.npz'
        if getattr(main,'last',None)!=orig:
            with np.load(orig) as d:
                arrays={k:d[k] for k in ('group','reach','pixels','action')}
            main.last=orig
        eligible=(arrays['group']==new['group']) & arrays['reach'].astype(bool) & (arrays['pixels']>=369)
        feasible=bool(np.any(eligible & ((arrays['action']>0).sum(1)+2<=budget)))
        arms.append((old_clear,new_clear,feasible))
    assert len(arms)==94
    actual=np.array(arms,dtype=bool)
    def parts(rows):return delta(*zip(*[(bool(e[0]),bool(e[1])) for e in rows]))
    report=dict(groups=94,geometrically_clear=parts(arms),per_house_in_frozen_order=[parts(house[h]) for h in json.loads((BASE/'view_design.json').read_text())['cohorts']['blind']],
                prior_frozen_library_contains_feasible_view=parts(actual[actual[:,2]]),
                no_original_library_witness=parts(actual[~actual[:,2]]),**seen,
                scope='Full same-budget physical path if inspection did not stop; ground truth visible mask at any executed step >=369. Actual first report audited separately. Independent feasible set uses original physically replayed candidate library; absence of such a witness does not prove inaccessibility.')
    (HERE/'route_geometry.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
