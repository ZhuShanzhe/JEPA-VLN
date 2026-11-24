"""Run an already frozen policy's own four choices in continuous Habitat independently."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def retake(project,asset,record,dst):
    sys.path.insert(0,str(project/'jepa_route_v12/expansion/hssd'))
    from collect_branches import ACT, WORDS, simulate, semantics, world_state
    from tree import visible
    root=project/'jepa_env_v13/simulator/trees/fresh'/record['house']
    folder=root/record['origin']
    pose=next(x for x in json.loads((root/'private_states.json').read_text()) if x['origin']==record['origin'])
    q=WORDS.index(record['word']); truth=json.loads((folder/'truth.json').read_text())['nodes']
    sim=simulate(asset,record['house']);found=set(); path='r';counts=[];ts=[];n_rgb=0
    steps=0
    try:
        lookup=semantics(sim,asset);agent=sim.get_agent(0)
        agent.set_state(world_state(pose['position'],pose['yaw']))
        def check(expect,expected_ids):
            nonlocal n_rgb
            obs=sim.get_sensor_observations(); ids=visible(obs,len(lookup))
            assert np.array_equal(expect,obs['rgb'][:,:,:3]) and ids==expected_ids
            n_rgb+=1;before=len(found); found.update(i for i in ids if lookup[i]==q)
            ts.extend([steps]*(len(found)-before))
        for choice in record['path']:
            one=truth[path]
            with np.load(folder/('node_'+path+'.npz')) as z:
                for t,a in enumerate((2,1,1,2),1):
                    agent.act(ACT[a]);steps+=1
                    if t<4:check(z['ctx'][t-1],one['scan'][str(t)])
                for t,a in enumerate(z['plans'][choice],1):
                    bump=agent.act(ACT[a]);steps+=1
                    assert bool(a==0 and bump)==bool(z['collided'][choice,t-1])
                    if t%6==0:check(z['walk_rgb'][choice,t//6-1],one['walk'][choice][str(t)])
            path+=str(choice);counts.append(len(found))
        assert counts==record['found'] and record['first_step']==(ts[0] if ts else None)
        assert record['second_step']==(ts[1] if len(ts)>1 else None)
    finally:sim.close()
    dst.write_text(json.dumps(dict(steps=steps,checked_rgb_and_ids=n_rgb,confirmed=counts,
                                   first_seen=ts[0] if ts else None,second_seen=ts[1] if len(ts)>1 else None)))


def verify(args):
    exp=args.project/'jepa_env_v13';d=exp/'simulator'
    rows={}
    with args.decisions.open() as src:
        for line in src:
            r=json.loads(line)
            if r['seed']==11 and r['method']==args.method:
                rows[r['house'],r['origin'],r['word']]=r
    houses=json.loads((d/'cohort.json').read_text())['fresh']
    work=d/'policy_checks'/args.method;work.mkdir(parents=True,exist_ok=True)
    def run(house):
        for p in sorted((d/'trees/fresh'/house).glob('*/episode.json')):
            m=json.loads(p.read_text())
            w=next((t['word'] for t in m['targets'] if t['stratum']=='multiple'),None)
            if w is not None:
                rec=rows[house,m['origin'],w]
                break
        else:raise ValueError('no multiple-instance query in house')
        part=work/(house+'.json')
        subprocess.run([str(args.hab),'-Bu',__file__,'one',str(args.project),str(args.asset),
                        json.dumps(rec),str(part)],check=True,env=dict(os.environ,MAGNUM_LOG='quiet',
                        HABITAT_SIM_LOG='quiet',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        return json.loads(part.read_text())
    with ThreadPoolExecutor(3) as pool: reports=list(pool.map(run,houses))
    out=dict(method=args.method,seed=11,query_selection='first eligible origin per house, first multiple target in frozen order',
             houses=len(houses),actions=sum(r['steps'] for r in reports),
             exact_rgb_and_id_sets=sum(r['checked_rgb_and_ids'] for r in reports),
             cursor_and_event_times_match=True,collision_flags_match=True,
             total_distinct_over_budgets=np.sum([r['confirmed'] for r in reports],axis=0).tolist())
    args.output.write_text(json.dumps(out,indent=2))


if __name__=='__main__':
    if sys.argv[1]=='one':
        retake(Path(sys.argv[2]),Path(sys.argv[3]),json.loads(sys.argv[4]),Path(sys.argv[5]))
    else:
        p=argparse.ArgumentParser()
        for k in ('project','asset','hab','decisions','output'):p.add_argument(k,type=Path)
        p.add_argument('--method',default='jepa_full')
        verify(p.parse_args())
