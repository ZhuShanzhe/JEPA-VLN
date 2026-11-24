"""Check causal, reversible and recoverable memory using only the live policy interface."""
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np


def run(module_dir, weights):
    sys.path.insert(0, str(module_dir))
    from v10_live import RouteMemory

    pos = [0., 0.]
    front = [1., 0.]
    left = [0.5, math.sqrt(3) / 2]
    right = [0.5, -math.sqrt(3) / 2]
    plans = [[1, 1] + [0] * 10, [2, 2] + [0] * 10]
    prior = [.2, .2]

    def decision(m, step, arm="full"):
        return np.asarray(m.rank(plans, pos, front, step, prior, prior, prior, arm=arm)["scores"])

    source = RouteMemory(weights)
    reset = decision(source, 0)
    source.observe(pos, 0, left, visual_score=.9)
    source.observe(pos, 1, right, visual_score=.05)
    left_clue = decision(source, 1)
    assert np.array_equal(source.features(plans, pos, front, 1), source.features(plans, pos, front, 1))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "state.json"
        source.save(path)
        changed, stable = RouteMemory(weights), RouteMemory(weights)
        for replica in (changed, stable):
            replica.load(path)
            assert np.array_equal(left_clue, decision(replica, 1))
            replica.observe(pos, 2, front)
        assert np.array_equal(decision(changed, 2), decision(stable, 2))
        for m, seen_left, seen_right in [(changed, .05, .9), (stable, .9, .05)]:
            m.observe(pos, 3, left, seen_left)
            m.observe(pos, 4, right, seen_right)
        changed_scores, stable_scores = decision(changed, 4), decision(stable, 4)
        inactive = decision(changed, 4, arm="fixed")
        assert np.array_equal(inactive, decision(stable, 4, arm="fixed"))
        changed.observe(pos, 4, right, .9)
        assert np.array_equal(changed_scores, decision(changed, 4))
        try:
            decision(changed, 3)
        except ValueError:
            pass
        else:
            raise AssertionError("scored a past decision with observations from its future")
        stale = changed.features(plans, pos, front, 244)
        fresh = changed.features(plans, pos, front, 4)
        assert np.max(abs(stale[:, 2])) < np.max(abs(fresh[:, 2]))
        changed.reset()
        assert np.array_equal(reset, decision(changed, 0))

    movement = RouteMemory(weights)
    straight_or_back = [[0] * 12, [1] * 6 + [0] * 6]
    movement.observe(pos, 0, front)
    movement.observe(pos, 1, front, collided=True)
    blocked = movement.features(straight_or_back, pos, front, 1)[:, 1]
    movement.observe([.25, 0.], 2, front)
    cleared = movement.features(straight_or_back, pos, front, 2)[:, 1]
    assert blocked[0] < cleared[0] and abs(cleared[0]) < 1e-12
    return dict(save_load_exact=True, identical_unobserved_change_prefix=True, reset_exact=True,
                duplicate_write_idempotent=True, future_history_refused=True,
                blocked_direction_reopens_after_actual_traversal=True,
                semantic_clue_peak_fresh=float(np.max(abs(fresh[:, 2]))),
                semantic_clue_peak_after_240_steps=float(np.max(abs(stale[:, 2]))),
                score_margin_left_minus_right_same_room=float(stable_scores[0]-stable_scores[1]),
                score_margin_left_minus_right_after_new_observations=float(changed_scores[0]-changed_scores[1]),
                synthetic_semantic_reversal_changes_top_choice=bool(np.argmax(changed_scores)!=np.argmax(stable_scores)),
                note="Constructed two-candidate interface test with supplied visual evidence and equal priors. Scores here do not assess perception or closed-loop navigation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("module_dir", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.module_dir, args.weights)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
