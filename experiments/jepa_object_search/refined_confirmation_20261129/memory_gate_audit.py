"""Explain provisional memories and the previously documented empty-box mistake after frozen testing."""
from collections import Counter
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/'reactive_view_20261129'


def main():
    provisional=Counter()
    best=[]
    prior_empty=[]
    for path in sorted((HERE/'private_paired/full').glob('*/*/decision.json')):
        house, source=path.parent.parent.name,path.parent.name
        out=json.loads(path.read_text())
        trace=out['trace']
        old=json.loads((ORIGINAL/'private_roll'/house/(source+'.json')).read_text())
        before=old['new_confirmation']
        if before is not None and old['events'][before]['intersection']==0:
            t=out['step']
            with np.load(HERE/'private_scoring'/house/(source+'.npz')) as gt:
                mask=np.unpackbits(gt['packed'][t]).reshape(192,192) if t is not None else None
            l,u,r,b=out['box'] if t is not None else (0,0,0,0)
            prior_empty.append(dict(initial_state=trace[0]['state'],prior_step=before,
                           new_step=t,new_intersection=None if mask is None else int(mask[u:b,l:r].sum()),
                           new_target_px=None if mask is None else int(mask.sum()),risk=trace[0]['contamination_risk'],
                           trust=trace[0]['p_initial']))
        if trace[0]['state']!='pending':
            continue
        with np.load(HERE/'private_inputs'/house/(source+'.npz')) as obs:
            odo=obs['odom'][:len(trace)]
        moved=np.linalg.norm(odo[:,:2],axis=1)>=.12
        tried=[z['p_relook'] for allowed,z in zip(moved,trace) if allowed and z['p_relook'] is not None]
        provisional['entered_pending']+=1
        provisional['at_least_one_displaced_frame']+=int(any(moved))
        provisional['displaced_with_nonempty_relook_score']+=int(bool(tried))
        provisional['improved_memory_accepted']=provisional.get('improved_memory_accepted',0)+int(trace[-1]['state']!='pending')
        if tried:best.append(float(max(tried)))
    report=dict(provisional=provisional, scores_when_available=sorted(best), prior_empty_box=prior_empty)
    (HERE/'memory_gate_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
