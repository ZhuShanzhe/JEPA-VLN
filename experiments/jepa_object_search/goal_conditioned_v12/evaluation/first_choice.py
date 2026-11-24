"""Independent, paired scoring of first-route decisions against held-out HSSD truth."""
import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np


ARMS = ("v12_jepa_crop", "v12_jepa", "v12_direct", "v11_jepa", "v11_direct")
SEEDS = (11, 29, 47)
CONTRASTS = (("v12_jepa_crop", "v11_jepa"), ("v12_jepa_crop", "v12_direct"),
             ("v12_jepa_crop", "v12_jepa"), ("v12_jepa", "v11_jepa"), ("v12_jepa", "v12_direct"),
             ("v12_direct", "v11_direct"))
CHOICE = ("hit", "number_distinct_instances", "fraction_of_room_instances",
          "uniform_choice_hit", "best_route_hit", "tied_high_score")
SWAPPED = ("hit_given_right_goal", "hit_after_exchanging_routes", "paired_goal_advantage",
           "both_goals_satisfied", "changed_route")


def outcomes(cache, split, raw):
    """Join goal names and candidate order to the separately stored instance counts."""
    result = {}
    action_shapes = set()
    files = sorted((cache / "actor").glob("*.npz"))
    if not files:
        raise ValueError(f"missing actor cache: {cache}")
    for path in files:
        with np.load(path, allow_pickle=False) as actor:
            house, origin = str(actor["house"]), str(actor["origin"])
            assert str(actor["split"]) == split
            actions = actor["actions"]
            assert actions.shape[0] == 3
            action_shapes.add(tuple(actions.shape))
            words = actor["words"].tolist()
        with np.load(cache / "truth" / path.name, allow_pickle=False) as truth:
            gains = truth["y"]
            if "house_instances" in truth:
                total = truth["house_instances"]
            else:
                # Some of the oldest caches omit the total inventory; the canonical
                # collector record for that exact origin includes it.
                with np.load(raw / house / f"{origin}.npz", allow_pickle=False) as original:
                    assert original["words"].tolist() == words
                    total = original["house_instances"]
            assert gains.shape == (3, len(words)) and total.shape == (len(words),)
            assert np.all(gains >= 0) and np.all(gains <= total[None, :])
            assert np.array_equal(gains, gains.astype(int))
            for idx, word in enumerate(words):
                key = (house, origin, word)
                if key in result:
                    raise ValueError(f"duplicate target key: {key}")
                result[key] = (gains[:, idx].astype(int), int(total[idx]))
    assert len(action_shapes) == 1
    return result, next(iter(action_shapes))[1]


def read_choices(path, truth, split):
    found = {}
    for row in map(json.loads, path.read_text().splitlines()):
        assert row["split"] == split
        key = (str(row["house"]), str(row["origin"]), str(row["query"]))
        first, seed, arm = row["first"], row["seed"], row["arm"]
        assert key in truth and seed in SEEDS and arm in ARMS
        score = np.array(row["score"])
        assert score.shape == (3,) and np.isfinite(score).all()
        assert first == np.argmax(score)
        final_key = (*key, seed, arm)
        if final_key in found:
            raise ValueError(f"duplicate prediction: {final_key}")
        found[final_key] = first, bool((score == score[first]).sum() > 1)
    assert len(found) == len(truth) * len(SEEDS) * len(ARMS)
    return found


def check_raw_panel(truth, raw, cohort, split):
    source_split = "test" if split in ("fresh", "fresh2") else split
    houses = set(json.loads(cohort.read_text())[source_split])
    receipts = {p.parent.name:p for p in raw.glob("*/receipt.json")}
    if set(receipts) != houses:
        raise ValueError("missing or unexpected frozen house receipts")
    keys = set()
    empty = 0
    for house, receipt in receipts.items():
        d = json.loads(receipt.read_text())
        assert d["house"] == house and d["split"] == source_split
        files = sorted(receipt.parent.glob("*.npz"))
        assert len(files) == d["origins"]
        empty += int(not files)
        n = 0
        for f in files:
            with np.load(f, allow_pickle=False) as z:
                assert str(z["house"]) == house
                words, y, total = z["words"].tolist(), z["y"], z["house_instances"]
                for q, w in enumerate(words):
                    key = (house, str(z["origin"]), w)
                    assert key not in keys
                    gains, count = truth[key]
                    assert np.array_equal(gains, y[:, q]) and count == total[q]
                    keys.add(key)
                n += len(words)
        assert n == d["queries"]
    if keys != set(truth):
        raise ValueError("cached and collected query sets differ")
    return dict(planned_houses=len(houses), finished_receipts=len(receipts), houses_without_eligible_queries=empty,
                cached_and_raw_query_ids_match=True, cached_and_raw_instance_counts_match=True)


def add(table, key, value):
    table[key].append(value)


def house_estimates(rows):
    # Each origin has the same weight despite differing numbers of target queries.
    by_house = defaultdict(list)
    for (house, _origin, _seed, arm), values in rows.items():
        by_house[house, arm].append(np.mean(values, axis=0))
    return {(house, arm): np.mean(values, axis=0)
            for (house, arm), values in by_house.items()}


def report(rows, labels):
    if not rows:
        return dict(houses=0, origins=0, unique_query_or_goal_pairs=0)
    sub = house_estimates(rows)
    houses = sorted({h for h, arm in sub})
    mean = {}
    for arm in ARMS:
        mean[arm] = dict(zip(labels, np.mean([sub[h, arm] for h in houses], axis=0).tolist()))
    comparisons = {}
    for candidate, reference in CONTRASTS:
        delta = np.stack([sub[h, candidate]-sub[h, reference] for h in houses])
        rng = np.random.default_rng(12041)
        draw = rng.integers(0, len(houses), size=(20000,len(houses)))
        ci = np.quantile(delta[draw].mean(1), [.025,.975], axis=0) if len(houses)>1 else None
        comparisons[f"{candidate}_minus_{reference}"] = {
            name: dict(delta=float(delta[:, j].mean()),
                       ci95=None if ci is None else ci[:, j].tolist(),
                       house_wins_ties_losses=[int((delta[:,j]>1e-10).sum()),
                                               int((abs(delta[:,j])<=1e-10).sum()),
                                               int((delta[:,j]<-1e-10).sum())])
            for j, name in enumerate(labels)}
    return dict(houses=len(houses), origins=len({(h,o) for h,o,_s,_a in rows}),
                unique_query_or_goal_pairs=sum(map(len,rows.values()))//(len(SEEDS)*len(ARMS)),
                mean=mean, comparisons=comparisons)


def run(cache, file, split, raw, cohort):
    truth, steps = outcomes(cache, split, raw)
    pred = read_choices(file, truth, split)
    conditions = defaultdict(lambda: defaultdict(list))
    goal_pairs = defaultdict(list)
    targets = defaultdict(list)
    for (house, origin, word), (gain, total) in truth.items():
        targets[house, origin].append(word)
        assert total >= 1
        positive = gain > 0
        groups = ("all", f"{int(positive.sum())}_profitable_routes")
        for arm in ARMS:
            for seed in SEEDS:
                first, tie = pred[house, origin, word, seed, arm]
                items = [positive[first], gain[first], gain[first]/total,
                         positive.mean(), positive.max(), tie]
                for group in groups:
                    add(conditions[group], (house, origin, seed, arm), items)
    for (house, origin), words in targets.items():
        for a, b in combinations(words, 2):
            good_a, good_b = (truth[house, origin, w][0] > 0 for w in (a, b))
            if not good_a.any() or not good_b.any() or (good_a & good_b).any():
                continue
            for arm in ARMS:
                for seed in SEEDS:
                    i,j = (pred[house, origin, w, seed, arm][0] for w in (a,b))
                    matched = (int(good_a[i])+int(good_b[j]))/2
                    exchanged = (int(good_a[j])+int(good_b[i]))/2
                    add(goal_pairs, (house, origin, seed, arm),
                        [matched,exchanged,matched-exchanged,good_a[i] and good_b[j],i!=j])
    return dict(split=split, paired_seeds=SEEDS, actions_per_route=steps,
                locked_predictions_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                aggregation="average goals or pairs at each origin, then origins and paired seeds in each house, then houses",
                uncertainty="20,000 house-paired bootstrap draws; intervals condition on these three seeds",
                choice={group:report(rows,CHOICE) for group,rows in sorted(conditions.items())},
                disjoint_targets=report(goal_pairs,SWAPPED), source_audit=check_raw_panel(truth,raw,cohort,split))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("choices", type=Path)
    ap.add_argument("split", choices=("fresh","fresh2","test","calibration"))
    ap.add_argument("out", type=Path)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--cohort", type=Path, required=True)
    args = ap.parse_args()
    if args.split == "fresh2":
        ARMS = ("rv_jepa", "rv_direct", "v11_jepa")
        CONTRASTS = (("rv_jepa","v11_jepa"), ("rv_jepa","rv_direct"), ("rv_direct","v11_jepa"))
    payload = run(args.cache, args.choices, args.split, args.raw, args.cohort)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
