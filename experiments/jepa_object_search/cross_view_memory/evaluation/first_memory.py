"""Replay instance encounters against an append-only first-sighting gallery."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def stable_order(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).digest()


def make_stream(rows, indices):
    # This is an encounter stream, not a robot trajectory. Interleave randomly
    # ordered views to avoid presenting all views of an object consecutively.
    ordered = sorted(indices, key=lambda i: stable_order(
        "first-memory-v1", rows[i]["instance"], rows[i]["view"]))
    return ordered


def replay(similarity, instances, threshold):
    gallery, seen, count = [], set(), Counter()
    for t, identity in enumerate(instances):
        first = identity not in seen
        count["first_encounters" if first else "revisits"] += 1
        seen.add(identity)
        if gallery:
            best = gallery[int(np.argmax(similarity[gallery, t]))]
            matched = float(similarity[best, t]) >= threshold
        else:
            matched = False
        if matched:
            if instances[best] == identity:
                count["correct_revisit_matches"] += 1
            else:
                count["wrong_merges"] += 1
                if first:
                    count["first_encounters_wrongly_merged"] += 1
        else:
            if any(instances[old] == identity for old in gallery):
                count["extra_records"] += 1
            gallery.append(t)
    count["instances_represented_at_end"] = len(set(instances[i] for i in gallery))
    count["records_at_end"] = len(gallery)
    return count


def evaluate(groups, method, threshold):
    by_house = {}
    for (house, _category), (instances, scores) in groups.items():
        by_house.setdefault(house, Counter()).update(replay(scores[method], instances, threshold))
    return by_house


def aggregate(by_house):
    c = sum(by_house.values(), Counter())
    return {k: c[k] for k in ("first_encounters", "revisits", "correct_revisit_matches",
                               "first_encounters_wrongly_merged", "wrong_merges", "extra_records",
                               "instances_represented_at_end", "records_at_end")}


def calibration_limits(groups, method):
    # Predeclared grid in validation-score quantiles. Calibrate in a gallery,
    # where a first sighting can be compared with more than one stored view.
    scores = np.concatenate([data[1][method][np.triu_indices(len(data[0]), k=1)]
                             for data in groups.values()])
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return 1.0
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 81))).tolist()
    candidates.append(float(np.nextafter(np.float64(np.max(scores)), np.inf)))
    feasible = []
    for threshold in candidates:
        total = aggregate(evaluate(groups, method, threshold))
        if total["first_encounters_wrongly_merged"] <= .02 * total["first_encounters"]:
            feasible.append((total["correct_revisit_matches"], -total["wrong_merges"], threshold))
    return max(feasible)[2]


@torch.inference_mode()
def prepare(rows, features, split, heads):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row["split"] == split:
            groups[row["house"], row["category"]].append(i)
    out = {}
    for name, group in groups.items():
        order = make_stream(rows, group)
        f = features[order]
        headings = np.asarray([rows[i]["camera_heading_deg"] for i in order])
        a, b = np.triu_indices(len(group), k=1)
        delta = torch.tensor((headings[b] - headings[a] + 180.) % 360. - 180., dtype=torch.float32)
        tables = {"frozen": (f @ f.T).numpy()}
        for tag, head in heads.items():
            batches = []
            for start in range(0, len(a), 256):
                s = slice(start, start + 256)
                logits, _ = head(f[a[s]], f[b[s]], delta[s])
                batches.append(logits.sigmoid().view(-1).numpy())
            table = np.eye(len(group), dtype=np.float32)
            table[a, b] = np.concatenate(batches)
            table[b, a] = table[a, b]
            tables[tag] = table
        out[name] = [list(rows[i]["instance"] for i in order), tables]
    return out


def differences(ja, other):
    houses = sorted(ja)
    assert houses == sorted(other)
    ix = np.random.default_rng(20261125).integers(0, len(houses), (5000, len(houses)))
    result = {}
    for num, den in (("correct_revisit_matches", "revisits"),
                     ("first_encounters_wrongly_merged", "first_encounters")):
        n = np.array([ja[h][num] - other[h][num] for h in houses])
        d = np.array([ja[h][den] for h in houses])
        diffs = n[ix].sum(axis=1) / d[ix].sum(axis=1)
        result[num] = {"difference_pp": 100 * n.sum() / d.sum(),
                       "house_bootstrap_95_pp": (100 * np.quantile(diffs, [.025,.975])).tolist()}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--validation-label", default="val")
    p.add_argument("--refiner", action="store_true")
    args = p.parse_args()
    model = args.root / "model"
    sys.path.insert(0, str(model))
    from reid_model import ViewHead
    rows_file = args.root / "data" / "manifest.jsonl"
    rows = [json.loads(line) for line in rows_file.read_text().splitlines()]
    cache = torch.load(model / "features.pt", map_location="cpu", weights_only=True)
    assert cache["manifest_sha256"] == hashlib.sha256(rows_file.read_bytes()).hexdigest()
    f = cache["features"].float()
    assert len(f) == len(rows) and torch.allclose(f.norm(dim=1), torch.ones(len(f)), atol=.001)
    heads = {}
    for tag in ("jepa", "direct"):
        ckpt = torch.load(model / (tag + ".pt"), map_location="cpu", weights_only=True)
        head = ViewHead()
        head.load_state_dict(ckpt["state_dict"])
        heads[tag] = head.eval()
    if args.refiner:
        from context_refiner import Refiner
        params = json.loads((args.root/"evaluation"/"refiner_parameters.json").read_text())
        heads["jepa_context_verifier"] = Refiner(heads["jepa"],params).eval()
    torch.set_num_threads(2)
    val = prepare(rows, f, args.validation_label, heads)
    test = prepare(rows, f, "test", heads)
    assert val and test and {h for h,c in val}.isdisjoint(h for h,c in test)
    methods = {}
    per_house = {}
    for tag in ("frozen", *heads):
        t = calibration_limits(val, tag)
        per_house[tag] = evaluate(test, tag, t)
        methods[tag] = {"threshold": t, "val": aggregate(evaluate(val, tag, t)),
                        "test": aggregate(per_house[tag])}
    candidates = [k for k in methods if k.startswith("jepa")]
    selected = max(candidates,key=lambda k:(methods[k]["val"]["correct_revisit_matches"],
                                             -methods[k]["val"]["wrong_merges"]))
    result = {"protocol": "first-memory-v1", "calibration_first_sighting_merge_limit": .02,
              "predictive_variant_selected_on_validation": selected,
              "validation_houses": len({h for h,c in val}), "test_houses": len({h for h,c in test}),
              "test_house_category_streams": len(test), "methods": methods,
              "paired_selected_minus_reference": {tag: differences(per_house[selected], per_house[tag])
                                                   for tag in ("direct", "frozen", "jepa")}}
    file = args.root / "evaluation" / "memory_results.json"
    file.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
