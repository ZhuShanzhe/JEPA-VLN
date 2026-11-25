"""Aggregate paired continuous HSSD V13 rollout logs at identical total action budgets."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def audit(project):
    root = project / 'jepa_env_v13'
    path = root/'online/fresh_horizon_decisions.jsonl'
    replay = json.loads((root/'simulator/horizon_policy_check.json').read_bytes())
    assert replay['cursor_and_event_times_match'] and replay['exact_rgb_and_id_sets'] == 336
    seeds = (11, 29, 47)
    methods = ('v11_baseline', 'direct_horizon', 'jepa_horizon')
    budgets = (28,56,84,112)
    records = defaultdict(dict)
    for line in path.read_bytes().splitlines():
        row = json.loads(line)
        if row['method'] not in methods or row['stratum'] != 'multiple':
            continue
        key = (row['house'], row['origin'], row['word'])
        pair = (row['method'], row['seed'])
        assert row['quota'] == 2 and row['count'] >= 2 and pair not in records[key]
        assert len(row['found']) == 4 and sorted(row['found']) == row['found']
        records[key][pair] = row['found']
    assert len(records)==171 and len({k[0] for k in records})==12
    assert all(len(v)==len(methods)*len(seeds) for v in records.values())
    def stat(values,b):
        return dict(distinct=sum(v[b] for v in values),
                    found_two=sum(v[b]>=2 for v in values),found_any=sum(v[b]>0 for v in values))
    result = {}
    for method in methods:
        per = {seed:{budget:stat([p[method,seed] for p in records.values()],b)
                     for b,budget in enumerate(budgets)} for seed in seeds}
        result[method] = dict(per_seed=per, seed_mean={n:{
            k:sum(per[s][n][k] for s in seeds)/3 for k in ('distinct','found_two','found_any')
        } for n in budgets})
    return dict(protocol='HSSD V13 four consecutive choices from real Habitat rollouts, seven RGB exposures per segment, 4 physical scans and 24 route actions per segment. Presence uses instance IDs (32px within 2m). No report action.',
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                multi_instance_queries=len(records), houses=12,seeds=seeds,results=result,
                independent_replay=dict(seed=replay['seed'],houses=12,actions=replay['actions'],
                                        compared_frames=336))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    doc=audit(args.project)
    args.output.write_text(json.dumps(doc,indent=2)+'\n')
    print({k:v['seed_mean'] for k,v in doc['results'].items()})
