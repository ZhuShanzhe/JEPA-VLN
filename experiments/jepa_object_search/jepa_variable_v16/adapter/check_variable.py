"""Deterministic contracts for variable native branches and set-independent ranking."""
import unittest
import numpy as np

from variable_routes import build_table
from variable_policy import score_routes


def rows(*plans):
    return [dict(id=f"f{i}", nominal_action_ids=plan) for i, plan in enumerate(plans)]


def table(plans, budget=24, original="f0"):
    entries, reason = build_table(rows(*plans), original, budget)
    return [e["id"] for e in entries], [e["plan"] for e in entries], reason


class CheckBranches(unittest.TestCase):
    def test_variable_counts_and_lengths(self):
        paths = [[1, 1], [2, 1, 1], [3, 1], [2, 2, 1], [3, 3, 1], [3, 3, 3, 1]]
        for n in (2, 3, 4, 5, 6):
            ids, plans, reason = table(paths[:n], original="f1")
            order = [1, 0, *range(2, n)]
            self.assertIsNone(reason)
            self.assertEqual(ids, [f"f{i}" for i in order])
            self.assertEqual(plans, [paths[i] for i in order])

    def test_only_real_spatial_branches(self):
        paths = [[1], [2, 3, 1], [1, 1, 1], [2, 3], [2, 1], [2, 1, 1], [2, 1, 3, 1]]
        ids, _, reason = table(paths)
        self.assertEqual(ids, ["f0", "f5", "f6"])
        self.assertIsNone(reason)

    def test_horizon_and_budget(self):
        paths = [[1] * 25, [1] * 24 + [2, 1], [1] * 21 + [3, 1, 1]]
        ids, plans, reason = table(paths, 100)
        self.assertEqual(ids, ["f0", "f2"])
        self.assertEqual([len(p) for p in plans], [24, 24])
        self.assertIsNone(reason)
        self.assertIsNotNone(table(paths, 22)[2])
        self.assertIsNotNone(table(paths, 0)[2])
        self.assertIsNone(table([[1], [2, 1]], 2)[2])

    def test_invalid_or_native_only(self):
        for paths in ([], [[2, 3], [1]], [[1]], [[1], [2, 1, 5]],
                      [[1], [2, True]], [[1], None]):
            self.assertIsNotNone(table(paths)[2])
        with self.assertRaises(ValueError):
            build_table([*rows([1]), *rows([2, 1])], "f0", 10)

    def test_scores_independent_of_alternatives_and_ties(self):
        base = [0.1, 0.2, 0.5, -0.1]
        rate = [0.2, 0.6, 0.1, 0.3]
        supply = [0.9, 1, 0.1, 0.3]
        def score(order):
            terms = [[v[i] for i in order] for v in (base, rate, supply)]
            return score_routes(*terms, 1., 80)["final_scores"]
        full = score(range(4))
        for order in ([2, 0], [3, 1, 0], [3, 2, 1, 0]):
            np.testing.assert_array_equal(score(order), [full[i] for i in order])
        self.assertEqual(score_routes([0, 0], [.1, .1], [1, 1], 1, 10)["index"], 0)


if __name__ == "__main__":
    unittest.main()
