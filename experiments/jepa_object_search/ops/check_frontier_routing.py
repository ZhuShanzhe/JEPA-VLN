"""Exercise the host boundary with held-out Habitat inputs and a reloaded pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from frontier_routing import choose_frontier


def run(base, checkpoint):
    sys.path.insert(0, str(base))
    import numpy as np
    from frontier_jepa.policy import load_ranker
    ranker = load_ranker(checkpoint)
    source = base / "a_exploration/paired_branches_v2"
    states = [json.loads(line) for line in (source / "index.jsonl").read_text().splitlines()]
    counts = dict(dev_states=0, candidates=0, proposed_different_from_test_anchor=0)
    for row in states:
        if row["split"] != "dev":
            continue
        folder = source / row["id"]
        with np.load(folder / "context.npz") as data:
            xy, plans = data["frontiers_xy"], data["planned_actions"]
        with np.load(folder / "features_vjepa21_b384.npz") as data:
            context = data["context_embedding"]
        ids = list(range(len(xy)))
        # A deterministic test anchor, not a decision made by MOST-Nav.
        anchor = ids[-1]
        def score(ordered):
            subset = list(ordered)
            return ranker.score(context, xy[subset], plans[subset], row["target"])
        shadow = choose_frontier("explore", anchor, ids, score)
        active = choose_frontier("explore", anchor, ids, score, apply=True)
        assert shadow.reason == "shadow" and active.reason == "apply"
        assert shadow.executed == anchor and active.executed == shadow.proposed in ids
        counts["dev_states"] += 1
        counts["candidates"] += len(ids)
        counts["proposed_different_from_test_anchor"] += int(active.executed != anchor)
    assert counts["dev_states"] == 48 and counts["candidates"] == 203
    def forbidden(_):
        raise AssertionError("The host action must bypass the ranker")
    for kind in ["report", "finish", "query", "visit", "unrecognized"]:
        assert choose_frontier(kind, None, None, forbidden, apply=True).reason == "host_action"
    assert choose_frontier("explore", 3, [3], forbidden, apply=True).reason == "single_candidate"
    assert choose_frontier("explore", 9, [2, 3], forbidden).reason == "invalid_host_candidates"
    tie = choose_frontier("explore", 3, [2, 3], lambda ids: [0.5, 0.5], apply=True)
    assert tie.executed == 3
    for invalid in [[0.5], [float("nan"), 1.0]]:
        invalid_choice = choose_frontier("explore", 3, [2, 3], lambda _: invalid, apply=True)
        assert invalid_choice.executed == 3 and invalid_choice.reason.startswith("scoring_error/")
    return dict(**counts, scope="offline routing interface; no MOST-Nav host or closed-loop score",
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), passed=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    checkpoint = args.root / "frontier_jepa/runs/hssd_v2_seed17_e5/predictive.pt"
    result = json.dumps(run(args.root, checkpoint), indent=2)
    if args.record:
        args.record.write_text(result + "\n")
    print(result)
