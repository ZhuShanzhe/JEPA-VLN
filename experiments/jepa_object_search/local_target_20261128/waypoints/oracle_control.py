"""Privileged physical reachability diagnostic, isolated from policy rollouts."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT/'local_target_20261128/eval_family'
PREVIOUS = ROOT/'approach_search_20261128/eval4'
sys.path[:0] = [str(FAMILY), str(PREVIOUS)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Trace:
    def __init__(self, sim):
        self.sim=sim
        self.positions=[sim.get_agent(0).state.position.tolist()]

    def step(self, action):
        obs=self.sim.step(action)
        self.positions.append(self.sim.get_agent(0).state.position.tolist())
        return obs

    def __getattr__(self, name):
        return getattr(self.sim,name)


def main():
    from physical_panel import DATA, hs, quat_from_coeffs, sensor_sim, measure
    from oracle_diagnostic import oracle, target_views
    panel_file=FAMILY/'panel.json'
    lock=json.loads((FAMILY/'panel_lock.json').read_text())
    for path in (panel_file,FAMILY/'physical_panel.py'):
        assert digest(path)==lock[str(path.relative_to(ROOT))]
    rows=[]
    for case in json.loads(panel_file.read_text()):
        assert case['budget']==300
        shard=DATA/'eval/hssdov/content'/(case['scene']+'.json.gz')
        assert digest(shard)==case['shard_sha256']
        ep=next(e for e in json.load(gzip.open(shard))['episodes'] if str(e['episode_id'])==str(case['id']))
        sim=sensor_sim(ep)
        try:
            sim.seed(1128)
            state=hs.AgentState()
            state.position=np.float32(ep['start_position']);state.rotation=quat_from_coeffs(ep['start_rotation'])
            sim.get_agent(0).set_state(state)
            trace=Trace(sim)
            result=oracle(trace,target_views(ep))
            scored=measure(sim,ep,trace.positions)
            assert len(trace.positions)==result['steps']+1
            assert scored['distinct_approached']==result['reached'] and scored['quota_met']==result['quota_met']
            rows.append(dict(scene=case['scene'], goal=case['goal'], episode=case['id'], controls=result,
                             rescored=scored, new_approaches=scored['distinct_approached']-scored['already_at_start']))
        finally:
            sim.close()
    report=dict(panel_sha256=digest(panel_file), map_privilege='full navmesh and ground-truth instance viewpoints',
                policy_access=False, max_steps=300, quota=2, oracle_sha256=digest(PREVIOUS/'oracle_diagnostic.py'),
                score_sha256=digest(FAMILY/'physical_panel.py'),rows=rows)
    output=FAMILY/'oracle_control.json'
    if output.exists():
        assert json.loads(output.read_text())==report
    else:
        with output.open('x') as f:
            json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps([(r['goal'],r['controls'],r['new_approaches']) for r in rows]))


if __name__=='__main__':
    main()
