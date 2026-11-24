"""Verify the complete online controller against frozen four-choice trajectories."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

from live_search import EnvironmentSearch


def check(project, all_origins=False):
    root=project/'jepa_env_v13'; tree=root/'simulator/trees/fresh'
    inputs={}
    for house in sorted(tree.iterdir()):
        manifests=sorted(house.glob('*/episode.json'))
        for m in manifests if all_origins else manifests[:1]:
            ep=json.loads(m.read_text())
            inputs[(ep['house'],ep['origin'])]=[v['word'] for v in ep['targets']]
    source=root/'online/fresh_layout_decisions.jsonl'
    frozen={}
    for line in source.read_text().splitlines():
        v=json.loads(line);key=v['house'],v['origin']
        if key in inputs and v['method'] in ('jepa_horizon','jepa_horizon_layout') and v['seed']==11:
            frozen[key,v['word'],v['method']]=v['path']
    metrics={}
    seed=11
    for method in ('jepa_horizon','jepa_horizon_layout'):
        controller=EnvironmentSearch(project,'chair.n.01',seed,horizon=True,layout=method.endswith('layout'))
        total=0;success=0;compared=0;first_mismatch=Counter()
        all_targets=sum(len(w) for w in inputs.values());houses=set()
        for (h,o),goals in inputs.items():
            files={}
            for goal in goals:
                controller.reset(goal);trace=[];p='r'
                for depth in range(4):
                    if p not in files:
                        with np.load(tree/h/o/('node_'+p+'.npz')) as z:
                            files[p]={k:z[k] for k in ('ctx','plans','walk_rgb','collided')}
                    node=files[p]
                    # Only the departure views and proposed actions enter the choice.
                    decision=controller.choose_after_scan(node['ctx'],node['plans']);j=decision['index']
                    trace.append(j)
                    # Supply physical feedback from the selected edge, even if that
                    # differs from the archived policy, before the next RGB scan.
                    controller.consume_walk(node['plans'][j],node['collided'][j],node['walk_rgb'][j])
                    p+=str(j)
                assert controller.turn==4 and controller.pending is None
                expected=frozen[(h,o),goal,method]
                difference=next((k for k in range(4) if trace[k]!=expected[k]),None)
                total+=1;success+=int(difference is None)
                compared+=4 if difference is None else difference+1
                if difference is not None:first_mismatch[difference+1]+=1
            houses.add(h)
        assert total==all_targets
        metrics[method]=dict(houses=len(houses),origins=len(inputs),trajectories=total,
            matching_entire_trajectory=success,physical_decisions=4*total,
            same_observation_decisions_compared=compared,first_mismatch_by_round=dict(first_mismatch))
    return dict(seed=seed,scope='all fresh origins' if all_origins else 'first lexicographic origin in every fresh house; every assigned target',
                private_reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),methods=metrics)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('project',type=Path);p.add_argument('--all-origins',action='store_true')
    a=p.parse_args();d=a.project/'jepa_env_v13/online';v=check(a.project,a.all_origins)
    out=d/'controller_check.json'
    out.write_text(json.dumps(v,indent=2))
    print(json.dumps(v),flush=True)
    if any(row['matching_entire_trajectory']!=row['trajectories'] for row in v['methods'].values()):
        raise SystemExit('some real-RGB decisions differed; inspect before treating cached and live as interchangeable')
