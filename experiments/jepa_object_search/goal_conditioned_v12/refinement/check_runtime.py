"""Verify live RGB inference against saved decisions on one junction per calibration house."""
import argparse
import json
from pathlib import Path
import numpy as np
from live_residual import BoundedSelector


def run(project):
    out=project/'jepa_route_v12/refinement'
    records={}
    for line in (out/'calibration_choices.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['seed']==11 and r['arm']=='rv_jepa':
            records[r['house'],r['origin'],r['query']]=r
    policy=BoundedSelector(project)
    mem=policy.old.remem(project)
    diffs=[]
    for h in sorted({k[0] for k in records}):
        path=sorted((project/'jepa_hidden_v11/hssd/calibration'/h).glob('*.npz'))[0]
        with np.load(path) as f:
            w=str(f['words'][0]);he=(f['headings']+180)%360-180
            rgb=f['ctx'][[int(abs(he-a).argmin()) for a in (-30,0,30)]]
            expected=records[h,str(f['origin']),w]
            score,_=policy.scores(rgb,w,f['actions'],mem,first_observation=True)
        assert int(score.argmax())==expected['first']
        diffs.append(float(abs(score-expected['score']).max()))
    assert len(diffs)==6 and max(diffs)<.003
    result=dict(arm='rv_jepa',seed=11,houses=len(diffs),choices_agree=len(diffs),max_abs_error=max(diffs))
    (out/'runtime_check.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('project',type=Path)
    run(p.parse_args().project)
