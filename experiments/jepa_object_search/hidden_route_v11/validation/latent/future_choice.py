"""Pair correct and cyclically mismatched future supervision on the frozen HSSD test."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired_analysis import SEEDS, summarize


def compare(path):
    paired = defaultdict(dict)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["source"] != "hssd" or row["arm"] not in ("jepa", "wrong_future"):
            continue
        assert row["split"] == "test"
        key = tuple(row[k] for k in ("mode", "house", "origin", "query", "seed"))
        assert row["arm"] not in paired[key]
        paired[key][row["arm"]] = row

    rows = defaultdict(list)
    for (mode, house, origin, _, seed), arms in paired.items():
        a, b = arms["jepa"], arms["wrong_future"]
        for k in ("gain", "horizon", "group"):
            assert a[k] == b[k]
        hit = np.asarray(a["gain"]) > 0
        assert len(hit) == 3 and seed in SEEDS
        if hit.any():
            rows[mode].append((house, origin, seed, [float(hit[a["first"]]), float(hit[b["first"]]), hit.mean()]))
    assert sorted(rows) == ["combined", "hssd"]
    result = {}
    for mode, decisions in rows.items():
        s = summarize(decisions)
        result[mode] = {k:s[k] for k in ("houses", "origins", "origin_queries", "delta", "ci95",
                                        "seed_deltas", "house_wins_ties_losses", "paired_house_signflip_p_two_sided")}
        result[mode]["jepa_wrong_future_uniform"] = s["jepa_direct_uniform"]
    return dict(unit="same as primary paired_analysis.summarize; all goal-origins with any profitable route",
                delta="JEPA minus cyclically mismatched future supervision", seeds=list(SEEDS), results=result)


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(compare(Path(sys.argv[1])), indent=2, allow_nan=False))
