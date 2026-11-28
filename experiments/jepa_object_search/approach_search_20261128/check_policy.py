import math
import unittest

import numpy as np

from map_views import candidates
from search_policy import PriorSearchMemory
from enhanced_search import PriorEvidenceMemory
from global_frontiers import frontier_plans
from motion_memory import MotionMemory


class GridFixture:
    res = .1
    origin = np.array([-4.,-4.])
    def __init__(self):
        self.obstacle = np.zeros((81, 81), dtype=bool)
        self.geometry_observed = np.ones_like(self.obstacle)
    def world_to_cell(self, point):
        return tuple(np.floor(np.array(point) / self.res + 40).astype(int))
    def in_bounds(self, cell):
        return all(0 <= n < 81 for n in cell)
    def traversable(self, cell):
        x, y = cell
        return self.in_bounds(cell) and not self.obstacle[y, x]


class SearchTest(unittest.TestCase):
    def test_prior_then_search_then_discover(self):
        rows = [dict(key=k, prior=p, position=(x, 0), heading=0) for k,p,x in
                (("near_wall", .85, 2), ("far_wall", .77, 4), ("other", .31, 6))]
        mem = PriorSearchMemory()
        for i in (0, 1, 2):
            idx, log = mem.choose(rows)
            self.assertEqual(idx, i)
            self.assertEqual(log["phase"], "prior_search" if i < 2 else "remaining_search")
            self.assertEqual(mem.choose(rows)[0], idx)
            mem.mark_view(rows[idx]["position"], rows[idx]["heading"])
        new = dict(key="new_wall", prior=.9, position=(8, 1), heading=0)
        self.assertEqual(mem.choose(rows+[new])[0], 3)

    def test_same_site_different_side_and_blocked(self):
        rows = [dict(key=str(i), prior=.75, position=(1, 1), heading=h) for i,h in enumerate((0, math.pi))]
        mem = PriorSearchMemory()
        mem.mark_view((1, 1), 0)
        self.assertEqual(mem.choose(rows)[0], 1)
        mem.mark_blocked((1, 1), math.pi)
        self.assertEqual(mem.choose(rows)[0], 0)
        for _ in range(12):
            mem.mark_view((1, 1), 0)
        self.assertEqual(mem.choose(rows)[0], 1)

    def test_geometry_and_budget(self):
        grid = GridFixture()
        grid.obstacle[:, 49] = True
        rows = candidates(grid, (0., 0., 0.), 1., [("forward", [1]*24)], 15)
        self.assertGreater(len(rows), 2)
        self.assertTrue(all(r["steps"] <= 15 and r["position"][0] < .9 for r in rows))
        self.assertTrue(any(r["source"] == "map_bend" for r in rows))

    def test_memory_excludes_blocked_steps_and_clears(self):
        memory, grid = MotionMemory(), GridFixture()
        memory.record((0.,0.,0.), 1., 0)
        guarded = memory.grid(grid,1.,1)
        self.assertTrue(guarded.traversable(grid.world_to_cell((0.,0.))))
        self.assertFalse(guarded.traversable(grid.world_to_cell((.23,0.))))
        self.assertTrue(memory.grid(grid, 1., 48).traversable(grid.world_to_cell((.23,0.))))

    def test_evidence_used_within_prior_group(self):
        rows=[dict(key=str(i), position=(i,0), heading=0, prior=p, evidence=e)
              for i,(p,e) in enumerate(((.8, .001),(.79,.28),(.2,.97)))]
        selected, log = PriorEvidenceMemory().choose(rows)
        self.assertEqual(selected,1)
        self.assertEqual(log["candidates"][1]["forecast_6_steps"],.28)

    def test_instruction_target_before_spatial_context(self):
        from pathlib import Path
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"prior_approach_20261128"))
        from target_words import requested_class
        classes={"clock","painting","cup","dining_table","table","lamp"}
        for text in ("Find two cups on the dining table", "Find all mugs on the table", "Find the cup next to the clock"):
            self.assertEqual(requested_class(text,classes),"cup")
        self.assertEqual(requested_class("Find all wall clocks",classes),"clock")
        self.assertIsNone(requested_class("Find clocks and lamps",classes))

    def test_frontiers_respect_motion_budget(self):
        grid = GridFixture()
        grid.geometry_observed[:,70:] = False
        rows = frontier_plans(grid,(0.,0.,0.),1.,18)
        self.assertTrue(rows)
        self.assertTrue(all(3<=len(plan)<=18 for _,plan in rows))


if __name__ == "__main__":
    unittest.main()
