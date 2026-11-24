"""Paired two-probe performance and missing-return bounds on frozen HSSD houses."""
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from paired_analysis import sign_flip


def paired(rows):
    by_origin = defaultdict(list)
    for (house, origin, _, seed), (jepa, direct) in rows.items():
        actual = np.array([jepa['union_gain'] > 0, direct['union_gain'] > 0], dtype=float)
        unobserved = np.array([not jepa['return_valid'], not direct['return_valid']]) * (1 - actual)
        by_origin[house, origin, seed].append([*actual, actual[0] - actual[1] - unobserved[1],
                                                actual[0] - actual[1] + unobserved[0]])
    by_seed = defaultdict(list)
    for (house, _, seed), vals in by_origin.items():
        by_seed[house, seed].append(np.mean(vals, axis=0))
    by_house = defaultdict(list)
    for (house, _), vals in by_seed.items():
        by_house[house].append(np.mean(vals, axis=0))
    vals = np.stack([np.mean(v, axis=0) for _, v in sorted(by_house.items())])
    deltas = vals[:, 0] - vals[:, 1]
    bootstrap = np.random.default_rng(711).choice(deltas, (20000, len(deltas))).mean(axis=1)
    return dict(houses=len(vals), seeded_pairs=len(rows), house_macro=vals.mean(0)[:2].tolist(),
                observed_delta=float(deltas.mean()), ci95=np.quantile(bootstrap, [.025,.975]).tolist(),
                possible_delta_with_missing_second=vals.mean(0)[2:].tolist(),
                house_signflip_p=sign_flip(deltas))


def run(project):
    receipts = [json.loads(f.read_text()) for f in sorted((project/'validation'/'returns').glob('*/receipt.json'))]
    fields = ('origins','branches','successful_returns','alternatives_from_return')
    result = dict(replay={name:sum(v[name] for v in receipts) for name in fields}, houses=len(receipts))
    result['replay']['max_first_frame_mae'] = max(v['max_first_frame_mae'] for v in receipts)
    result['replay']['max_origin_rgb_mae'] = max(v['max_origin_rgb_mae'] for v in receipts)
    result['replay']['first_gain_reproduced_all_houses'] = all(v['exact_first_gain'] for v in receipts)
    for mode in ('hssd','combined'):
        table = defaultdict(dict)
        for line in (project/'models'/'physical_returns.jsonl').read_text().splitlines():
            r = json.loads(line)
            if r['mode'] == mode and r['arm'] in ('jepa','direct'):
                key = r['house'],r['origin'],r['query'],r['seed']
                assert r['arm'] not in table[key]
                table[key][r['arm']] = r
        rows = {k:(v['jepa'],v['direct']) for k,v in table.items()}
        complete = {k:v for k,v in rows.items() if all(x['return_valid'] for x in v)}
        result[mode] = dict(confirmed=paired(rows), both_returns_valid=paired(complete),
                            interrupted_per_arm=[sum(not v[i]['return_valid'] for v in rows.values()) for i in (0,1)])
    return result


if __name__ == '__main__':
    root, dst = map(Path, sys.argv[1:])
    dst.write_text(json.dumps(run(root), indent=2, allow_nan=False))
