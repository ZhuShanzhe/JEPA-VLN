"""Explain the frozen proposal filters on every previously verified target-box event."""
from collections import Counter
import json
import math
from unittest.mock import patch

import numpy as np
import probe_branches as probe
from routes import _Map, _primitives, _end, _centered_evidence, observation_routes, TURN


def reject(online, end, anchor):
    dist = math.dist(end[:2], anchor)
    delta = math.remainder(math.atan2(anchor[1]-end[1], anchor[0]-end[0])-end[2], math.tau)
    if abs(delta) >= math.pi/4:
        return 'outside_camera_45_degree_half_angle'
    if dist < .30:
        return 'anchor_too_close'
    if _centered_evidence(online, end, anchor) <= 0:
        return 'observed_obstacle_in_sightline'
    return 'positive_view'


def diagnose(ev, public, grid, pose, seen, scan):
    geometry, receipt = scan.packet([scan.frames[j]['rgb'] for j in (1,0,3)],scan.frames[0]['depth'],pose)
    assert receipt == ev['physical_scan']
    cue = dict(category=public['goal'], view_index=ev['cue_view_index'], bbox_norm=ev['cue_bbox_norm'],
               candidate_id=ev['cue_candidate_id'])
    anchors, _, _ = probe.located_cue(cue,[scan.frames[j]['rgb'] for j in (1,0,3)],grid,pose,1.,geometry,90.)
    anchor = anchors[0]['anchor_xy']
    assert np.allclose(anchor,ev['cue_anchor_xy'],atol=1e-6)
    info = dict(goal=public['goal'], ordinal=public['sample'], house=public['house'], gate=public['gate'],
                source_pixels=public['initial_cue_pixels'], deployed_choices=public['choices'])
    if public['gate']=='blocked_or_no_regional_plan':
        info.update(preceding_contact=True)
        return info
    budget = 300-ev['step']
    n = min(budget,4)
    plan = ev['actions'][:n]
    proposal = observation_routes(grid,pose,1., mode='verify',region_actions=plan,budget=budget,
                                  observed_surfaces=seen,anchor_xy=anchor)
    assert len(proposal['candidates'])==public['choices']
    online = _Map(grid,1.)
    def key(p): return round(p[0]/.04), round(p[1]/.04), round(p[2]/TURN)%12
    prior = {key(_end(online,pose,plan))}
    counts = Counter()
    for moves in _primitives(n):
        end = _end(online,pose,moves)
        if end is None:
            counts['forward_path_fails_observed_free_clearance'] += 1
            continue
        if key(end) in prior:
            counts['same_endpoint_and_heading'] += 1
            continue
        prior.add(key(end))
        counts[reject(online,end,anchor)] += 1
    info.update(prefix_horizon=n, over_horizon_unenumerated=len(list(_primitives(4)))-sum(counts.values()),
                trial_plans=sum(counts.values()), first_filter_counts=dict(counts),
                positive_sector_pruned=counts['positive_view']-(len(proposal['candidates'])-1),
                anchor_range_m=round(math.dist(pose[:2],anchor),2),
                regional_control=reject(online,_end(online,pose,plan),anchor),
                cue_close_visible_on_any_alternative=public.get('best_same_cue_close_visible'),
                model_collision=public['model']['collision'])
    return info


def main():
    summary=probe.read(probe.HERE/'probe_results.json')
    rows=[r for r in summary['rows'] if r['inspection_valid']]
    assert len(rows)==4
    diagnostics=[]
    panel=probe.read(probe.FAMILY/'panel.json')
    for house in sorted({r['house'] for r in rows}):
        idx=house-1
        expected={r['sample']:r for r in rows if r['house']==house}
        facts={}
        def select(events):
            subset=[e for e in events if e.get('cue_selected') and e['anchors'] and not e['sites']]
            for ordinal,public in expected.items():
                ev=subset[ordinal-1]
                facts[ev['step']]=public
            return {e['step']:(e,'verify') for e in subset if e['step'] in facts}
        def snapshot(sim,scan,grid,seen,pose,e,mode,goal,channel,blocked):
            public=facts[e['step']]
            assert blocked==(public['gate']=='blocked_or_no_regional_plan')
            diagnostics.append(diagnose(e,public,grid,pose,seen,scan))
            return dict(gate=public['gate'],outcomes=[None]*public['choices'],selected='audit')
        with patch.object(probe,'select',select), patch.object(probe,'snapshot',snapshot):
            probe.run_house(idx,panel[idx],None)
    assert len(diagnostics)==4
    report=dict(cases=diagnostics,source_hashes={k:probe.digest(probe.HERE/k) for k in (
        'routes.py','probe_branches.py','diagnose_verified.py','probe_results.json')},
        procedure='Rebuild observed maps and capture checks from archived actions. Diagnoses use the deployed '
                  'proposal geometry and same source-verified anchors. Positive proposals may still collide in Habitat.')
    (probe.HERE/'verified_filters.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
