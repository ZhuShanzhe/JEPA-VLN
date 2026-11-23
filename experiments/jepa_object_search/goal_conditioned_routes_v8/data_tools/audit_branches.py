"""Audit whole-house partitions, physically executed actions and teacher isolation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def audit(root):
    replay = json.loads((root / "replay.json").read_text())
    seen_houses, seen_ids = set(), set()
    query_order, action_library = None, None
    summary = {}
    for split, expected, per_house in (("train", 12, 6), ("calibration", 4, 4), ("holdout", 4, 4)):
        rows = json.loads((root / (split + ".index.json")).read_text())
        houses, metric, alignment = {}, {}, []
        ids = [row["sid"] for row in rows]
        assert len(ids) == len(set(ids)) == expected * per_house and not seen_ids.intersection(ids)
        seen_ids.update(ids)
        class_counts = {}
        for row in rows:
            house = row["house"]
            assert row["split"] == split
            houses[house] = houses.get(house, 0) + 1
            alignment.append(row["baseline_rgb_mae"])
            with np.load(row["observed"]) as obs, np.load(row["observed_features"]) as enc:
                assert set(obs.files) == {"actions", "context_rgb", "yaw"}
                assert set(enc.files) == {"actions", "ctx", "yaw"}
                actions = obs["actions"]
                assert actions.shape == (9, 24) and len(np.unique(actions, axis=0)) == 9
                assert np.array_equal(actions, enc["actions"])
                assert np.isfinite(enc["ctx"]).all() and enc["ctx"].shape == (12, 3, 1024)
                if action_library is None:
                    action_library = actions.tolist()
                assert actions.tolist() == action_library
            if split == "holdout":
                assert not Path(row["features"]).exists(), "held-out future features unexpectedly cached"
            else:
                assert Path(row["features"]).is_file()
            with np.load(row["labels"]) as lab:
                query = lab["queries"].tolist()
                if query_order is None:
                    query_order = query
                assert query == query_order and len(set(query)) == 16
                present = lab["query_present"].astype(bool)
                gain = lab["gain"]
                assert gain.shape == (9, 3, 16)
                assert np.all(gain >= 0) and np.all(np.diff(gain, axis=1) >= 0)
                assert np.all(gain[:, :, ~present] == 0)
                for prefix in ("ctx", "future"):
                    grid, near = lab[prefix + "_grid"], lab[prefix + "_near_grid"]
                    assert np.all(near <= grid)
                    bins = np.bincount(np.arange(256) * 7 // 256)
                    assert np.all(grid.sum(axis=-3) <= bins[:, None] * bins[None, :])
                delta = np.diff(lab["actual_progress"], axis=1, prepend=0)
                assert np.all(np.isfinite(delta)) and np.all(delta >= -1e-5)
                assert np.all(delta[actions != 0] < 1e-5)
                active = gain[:, -1, :].max(axis=0) > gain[:, -1, :].min(axis=0)
                sides = ((1, 5), (2, 6), (3, 7), (4, 8))
                reversals = 0
                for left, right in sides:
                    diff = gain[left, -1] - gain[right, -1]
                    reversals += int(np.any(diff > 0) and np.any(diff < 0))
                for key, val in dict(
                    existing_state_queries=present.sum(),
                    state_queries_with_candidate_difference_at_24=active.sum(),
                    states_with_candidate_difference=active.any(),
                    states_with_opposite_target_preferences_on_symmetric_plans=reversals > 0,
                    symmetric_plan_pairs_with_opposite_target_preferences=reversals,
                    branches_with_any_new_reportable_instance=np.any(gain[:, -1] > 0, axis=1).sum(),
                    blocked_forward_commands=lab["collisions"][actions == 0].sum(),
                    forward_commands=(actions == 0).sum(),
                ).items():
                    metric[key] = metric.get(key, 0) + int(val)
                for q, avail, info in zip(query, present, active):
                    c = class_counts.setdefault(q, [0, 0])
                    c[0] += int(avail)
                    c[1] += int(info)
        assert len(houses) == expected and set(houses.values()) == {per_house}
        physical = replay[split]
        assert physical["states"] == len(rows) and physical["actions"] == len(rows) * 9 * 24
        assert physical["max_horizontal_step_m"] <= .251 and physical["max_replay_difference_m"] <= 1e-4
        assert not (seen_houses & set(houses))
        seen_houses.update(houses)
        summary[split] = dict(
            houses=expected, starts=len(rows), branches=len(rows) * 9,
            max_initial_rgb_mae_8bit=float(max(alignment)),
            median_initial_rgb_mae_8bit=float(np.median(alignment)),
            state_ids_sha256=digest(sorted(ids)), independent_physics_replay=physical, **metric,
            class_present_and_informative_state_counts=class_counts,
        )
    return dict(splits=summary, action_library_sha256=digest(action_library),
                class_order_sha256=digest(query_order), horizons=[4, 12, 24], observation_keys=["context_rgb", "actions", "yaw"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
