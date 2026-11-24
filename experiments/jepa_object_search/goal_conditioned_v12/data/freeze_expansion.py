"""Reproduce both V12 house allocations from the frozen V11 cohort."""
import argparse
import hashlib
import json
from pathlib import Path


def split(old):
    used = {house.split('_')[0] for group in ('protected','train','calibration','test') for house in old[group]}
    available = [h for h in old['fresh_candidates'] if h.split('_')[0] not in used]
    if len(available)!=len({h.split('_')[0] for h in available}):
        raise ValueError('house variants in the sampling frame')
    def order(homes,salt):
        return sorted(homes,key=lambda h: hashlib.sha256((salt+h).encode()).hexdigest())
    a=order(available,'search-v12-fresh-salt|')[:16]
    other=order([h for h in available if h not in a],'stage-b-route-dev|')
    return dict(stage_a_evaluation=a,stage_b_training=other[:12],stage_b_evaluation=other[12:24],
                reserved_count=len(other[24:]))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('v11_cohort',type=Path)
    args=parser.parse_args()
    print(json.dumps(split(json.loads(args.v11_cohort.read_text())),indent=2))
