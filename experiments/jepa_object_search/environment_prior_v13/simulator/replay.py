"""Independently retake one fixed, continuous four-choice episode path from every origin."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from tree import seed


def replay(project,asset,split,house):
    from collect_branches import ACT, WORDS, simulate, semantics, world_state
    d=project/'jepa_env_v13/simulator/trees'/split/house
    starts=json.loads((d/'private_states.json').read_text())
    sim=simulate(asset,house)
    totals=dict(roots=0,nodes=0,movements=0,rgb_checked=0,id_sets_checked=0,
                exact_rgb=0,max_rgb_mae=0.,collision_mismatch=0,id_set_mismatch=0)
    try:
        lookup=semantics(sim,asset)
        agent=sim.get_agent(0)
        for item in starts:
            origin=item['origin'];folder=d/origin
            truth=json.loads((folder/'truth.json').read_text())['nodes']
            path='r'
            agent.set_state(world_state(item['position'],item['yaw']))
            def check(expected,instance_ids):
                obs=sim.get_sensor_observations(); sem=obs['ids'];depth=obs['depth']
                good=(depth>0)&(depth<=2.)&np.isfinite(depth)&(sem>0)&(sem<len(lookup))
                ids=np.flatnonzero(np.bincount(sem[good],minlength=len(lookup))>=32).tolist()
                totals['id_sets_checked']+=1;totals['id_set_mismatch']+=ids!=instance_ids
                diff=np.abs(expected.astype('int16')-obs['rgb'][:,:,:3].astype('int16'))
                totals['rgb_checked']+=1;totals['exact_rgb']+=int(not np.any(diff))
                totals['max_rgb_mae']=max(totals['max_rgb_mae'],float(diff.mean()))
            for k in range(4):
                choice=seed('independent_replay13',house,origin,k)%3;one=truth[path]
                with np.load(folder/('node_'+path+'.npz')) as z:
                    for t,c in enumerate((2,1,1,2),1):
                        agent.act(ACT[c])
                        if t<4:check(z['ctx'][t-1],one['scan'][str(t)])
                    frames=z['walk_rgb'][choice];coll=z['collided'][choice]
                    for t,c in enumerate(z['plans'][choice],1):
                        b=agent.act(ACT[c])
                        totals['collision_mismatch']+=int(bool(c==0 and b)!=coll[t-1])
                        if t%6==0:check(frames[t//6-1],one['walk'][choice][str(t)])
                totals['movements']+=28; totals['nodes']+=1;path+=str(choice)
            totals['roots']+=1
    finally:sim.close()
    out=project/'jepa_env_v13/simulator/replay'/split;out.mkdir(parents=True,exist_ok=True)
    (out/(house+'.json')).write_text(json.dumps(totals))
    if totals['collision_mismatch'] or totals['id_set_mismatch'] or totals['max_rgb_mae']>2:
        raise RuntimeError('replay mismatch')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('project',type=Path);ap.add_argument('asset',type=Path)
    ap.add_argument('split');ap.add_argument('house');a=ap.parse_args()
    sys.path.insert(0,str(a.project/'jepa_route_v12/expansion/hssd'))
    replay(a.project,a.asset,a.split,a.house)
