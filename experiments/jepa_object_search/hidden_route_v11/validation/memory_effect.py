"""Hold first decisions fixed and compare physical second probes with and without memory updates."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def split_records(path):
    records = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get('source', 'hssd') != 'hssd' or row['arm'] not in ('jepa', 'direct'):
            continue
        key = tuple(row[k] for k in ('house', 'origin', 'query', 'seed', 'mode', 'arm'))
        if key in records:
            raise ValueError('Repeated policy decision')
        records[key] = row
    return records


def summarize(rows):
    by_origin = defaultdict(list)
    for house, origin, seed, values in rows:
        by_origin[house, origin, seed].append(values)
    by_seed = defaultdict(list)
    for (house, _, seed), group in by_origin.items():
        by_seed[house, seed].append(np.mean(group, axis=0))
    by_house = defaultdict(list)
    for (house, _), group in by_seed.items():
        by_house[house].append(np.mean(group, axis=0))
    values = np.stack([np.mean(x, axis=0) for x in by_house.values()])
    mean = values.mean(0)
    summary = dict(houses=len(values), distinct_origins=len({(h,o) for h,o,_,_ in rows}),
                   seeded_queries=len(rows), initial_hit=float(mean[0]), valid_return=float(mean[1]))
    metrics = ('cumulative_confirmation', 'any_additional_unique_instance',
               'additional_unique_instance_count', 'first_miss_rescued', 'cumulative_unique_instance_count')
    for i, name in enumerate(metrics):
        a, b = 3 + 2*i, 4 + 2*i
        diff = values[:,a] - values[:,b]
        boots = np.random.default_rng(711).choice(diff, (20000, len(diff))).mean(axis=1)
        summary[name] = dict(updated=float(mean[a]), initial_ranking=float(mean[b]), delta=float(diff.mean()),
                             house_bootstrap_95=np.quantile(boots,[.025,.975]).tolist())
    return summary, float(mean[2])


def run(project):
    files = [project/'models'/name for name in ('test_predictions.jsonl','physical_returns.jsonl')]
    scores, decisions = [split_records(p) for p in files]
    assert scores.keys() == decisions.keys()
    origin_keys = defaultdict(list)
    for key in scores:
        origin_keys[key[:2]].append(key)
    groups = defaultdict(list)
    for (house, origin), keys in sorted(origin_keys.items()):
        with np.load(project/'validation'/'returns_truth'/house/(origin+'.npz')) as truth:
            queries = truth['words'].tolist()
            first_y, union_y = truth['first_y'], truth['union_y']
        for key in keys:
            _, _, word, seed, mode, arm = key
            start, actual = scores[key], decisions[key]
            q = queries.index(word)
            sc = np.array(start['score'], dtype='float64')
            i = actual['first']
            assert i == start['first'] == int(np.argmax(sc)) and sc.shape == (3,)
            first = int(first_y[i,q])
            assert first == actual['first_gain'] == start['gain'][i]
            sc[i] = -np.inf
            static = int(np.argmax(sc))
            valid = actual['return_valid']
            chosen = actual['second']
            if valid:
                assert chosen is not None and chosen != i
                unions = [int(union_y[i, k, q]) for k in (chosen, static)]
                assert min(unions) >= first
            else:
                assert chosen is None and np.all(union_y[i,:,q] == -1)
                unions = [first, first]
            assert unions[0] == actual['union_gain']
            a,b = unions
            fields = [first>0, valid, valid and chosen!=static, a>0, b>0,
                      a>first, b>first, a-first, b-first, first==0 and a>0, first==0 and b>0, a,b]
            groups[mode,arm].append((house,origin,seed,fields))
    tables = {}
    for (mode,arm), group in sorted(groups.items()):
        valid = [r for r in group if r[-1][1]]
        all_stats,_ = summarize(group)
        valid_stats,switch = summarize(valid)
        tables[mode+'|'+arm] = dict(invalid_return_rows=len(group)-len(valid),
            confirmed_all=all_stats, valid_return_only=valid_stats,
            second_choice_switch_house_macro_valid=switch,
            second_choice_switch_raw_valid=sum(int(r[-1][2]) for r in valid))
    return dict(comparison='Stored real outbound-and-return memory choice vs best remaining initial score including initial memory; same model and identical first action plan.',
                observations='Goal opportunities sampled at four fixed physical frames on each outward search leg. Union counts deduplicate simulator instance IDs across legs. Physical return imagery updates memory only.',
                statistics='Queries within origin, origins per house and seed, seeds within house, houses equally weighted. Delta is updated minus initial ranking. House bootstrap 20000, seed 711. Exploratory comparisons.',
                denominators='Valid-return panel omits shared interruptions; confirmed-all retains only first-leg findings for shared interruptions. The switch rate uses valid returns. First-miss rescued uses all rows of the panel as denominator.',
                interpretation='Measures the joint contribution of cue, collision, time, pose and novelty updates to the second decision given the same first decision. No isolated semantic-memory effect or whole-navigation success rate.',
                private_input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, tables=tables)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('project',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    args.output.write_text(json.dumps(run(args.project),indent=2,allow_nan=False)+'\n')
