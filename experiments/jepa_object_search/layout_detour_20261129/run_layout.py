"""Real physical rollout of frozen local requests; true instance masks are logged only after action."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
REACTIVE=HERE.parent/'reactive_view_20261129'
BASE=HERE.parent/'new_view_20261129'
sys.path[:0]=[str(REACTIVE),str(BASE)]
import rollout
from sample_views import simulator,inventory,ASSETS

class Worker(rollout.Worker):
    def __init__(self,folder):
        self.packet=folder/'route_inputs.npz'
        self.log=(folder/'inference.log').open('a')
        env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('LD_LIBRARY_PATH',None)
        py=env.get('JEPA_POLICY_PYTHON','/root/autodl-tmp/jepa_instance_bench_20261016/policy_env/bin/python')
        self.proc=subprocess.Popen([py,'-u',str(HERE/'layout_worker.py'),str(self.packet)],stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,stderr=self.log,text=True,env=env)

def run(i):
    os.umask(0o077)
    manifest=json.loads((HERE/'freeze.json').read_text())
    for name,digest in manifest['sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    house=json.loads((BASE/'view_design.json').read_text())['cohorts']['blind'][i]
    store=HERE/'private_motion'/house;store.mkdir(parents=True,exist_ok=True)
    original=BASE/'private_render_blind60'/('blind_'+house)
    choices=json.loads((BASE/'private_evaluation/blind_choices.json').read_text())
    with np.load(original/'controls.npz') as f:
        d={k:f[k] for k in ('group','kind','source_index','budget','action','pixels','reach','first_blocked')}
    with np.load(BASE/'private_data'/('blind_'+house+'.npz')) as f:ctx=f['context']
    sim=simulator(house);worker=Worker(store)
    try:
        inventory(sim,ASSETS)
        for row in (r for r in choices if r['house']==house):
            name=str(row['source'])+'.json'
            baseline=json.loads((REACTIVE/'private_roll'/house/name).read_text())
            if not baseline['focus'] or (store/name).exists():continue
            idx=int(np.flatnonzero(d['group']==row['group'])[0])
            result=rollout.one(sim,worker,store,original,d,ctx[idx],row,'route')
            assert result['focus'] and result['budget']==baseline['budget']
            tmp=store/(name+'.partial');tmp.write_text(json.dumps(result,allow_nan=False));tmp.replace(store/name)
            print(house,name,len(result['events'])-1,flush=True)
    finally:worker.close();sim.close()
if __name__=='__main__':run(int(sys.argv[1]))
