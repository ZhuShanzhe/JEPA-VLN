"""Compare fresh RGB inference against stored first decisions in each calibration house."""
from collections import defaultdict
import json
from pathlib import Path
import argparse

import numpy as np
from live_routes import LiveRoutes


def run(project):
    out = project/'jepa_route_v12'/'models'
    policy=LiveRoutes(project)
    records = defaultdict(dict)
    for line in (out/'calibration_choices.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['arm']==policy.arm and r['seed']==11:
            records[r['house']][r['origin'],r['query']] = r
    memory=policy.old.remem(project)
    hit=0
    maximum=0.
    for house in sorted(records):
        file=sorted((project/'jepa_hidden_v11'/'hssd'/'calibration'/house).glob('*.npz'))[0]
        with np.load(file) as f:
            target=str(f['words'][0])
            angles=(np.array(f['headings'])+180)%360-180
            view=f['ctx'][[int(np.argmin(np.abs(angles-a))) for a in (-30,0,30)]]
            scores,_=policy.scores(view,target,f['actions'],memory,first_observation=True)
            known=records[house][str(f['origin']),target]
        maximum=max(maximum,float(np.abs(scores-known['score']).max()))
        hit+=int(int(scores.argmax())==known['first'])
    if hit!=len(records) or maximum>.003:
        raise AssertionError(f'cached/live mismatch: {hit} agreements, max score drift {maximum}')
    return dict(arm=policy.arm,samples=len(records),rooms=len(records),exact_choice=hit,max_score_error=maximum,
                source='one calibration junction per house; fresh RGB, text, plans and reset episode memory')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('project',type=Path)
    args=ap.parse_args()
    result=run(args.project)
    print(result)
    (args.project/'jepa_route_v12'/'models'/'live_check.json').write_text(json.dumps(result,indent=2))
