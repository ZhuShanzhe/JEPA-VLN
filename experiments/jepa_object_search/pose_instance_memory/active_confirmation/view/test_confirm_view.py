"""Production policy tests with prerecorded registration and injected observation receipts."""
import math
import unittest
from unittest.mock import Mock

import numpy as np
from agents.navigator import OccupancyGrid
from benchmark_api import Action
from confirm_view import ConfirmViewMixin, FORWARD, LEFT, choose_view, pose_after, return_to_reference
from repair_agent import RepairFrontier
import test_chain


class Policy(ConfirmViewMixin, RepairFrontier):
    pass


class ConfirmViewTests(unittest.TestCase):
    def setUp(self):
        f = test_chain.ReportChainTests()
        f.setUp()
        self.fixtures = f
        self.agent = a = f.make_agent(Policy)
        self.node = f.register(a, 3, (1.30, 0., .5))
        free = np.ones((45,45), dtype=bool)
        a._frontier_grid = OccupancyGrid(.1, (-2.,-2.), free, ~free)
        a._references = lambda n: [(3, object())]
        a._report_gate = Mock(return_value=(True, "localized_same_instance"))
        a._last_motion_failed = False
        a._category = lambda o: "painting"
        self.pose = (0.,0.,0.)
        a._estimated_current_pose = lambda: self.pose

    def next(self, step=20, previous=None):
        return self.agent._confirm_view_step(self.node, self.fixtures.observation(step, previous))

    def test_approach_recheck_dispatch_then_observed_report(self):
        step, previous = 20, None
        while True:
            action, kind = self.next(step, previous)
            if kind == "confirm_view_report":
                self.assertEqual(action, int(Action.TARGET_FOUND))
                break
            self.pose = pose_after(self.pose, action, 1.)
            previous = action
            step += 1
            self.assertLessEqual(step, 32)
        self.assertGreater(step, 20)
        self.assertLessEqual(math.dist(self.pose[:2], self.node.point[:2]), .70)
        self.assertFalse(self.node.reported)
        self.agent._settle_report(self.fixtures.observation(step+1, action))
        self.assertTrue(self.node.reported)
        self.assertIsNone(self.next(step+2, action))

    def test_stalled_forward_leaves_clue_open_and_cools_down(self):
        action, _ = self.next()
        self.assertEqual(action, FORWARD)
        self.agent._last_motion_failed = True
        self.assertIsNone(self.next(21, action))
        self.assertIsNone(self.next(22, action))
        self.assertFalse(self.node.reported)
        self.assertIsNone(self.agent._confirm_view_state)
        self.assertTrue(any(c.kwargs.get("reason") == "forward_stalled"
                            for c in self.agent._note.call_args_list))

    def test_command_not_executed(self):
        self.assertEqual(self.next()[0], FORWARD)
        self.assertIsNone(self.next(21, LEFT))
        self.assertIsNone(self.agent._pending_report)

    def test_arrived_without_live_match_does_not_claim_target(self):
        self.agent._report_gate.return_value = False, "other_instance_explains_same_live_region"
        reply, step = self.next(), 20
        while reply:
            action, _ = reply
            self.pose = pose_after(self.pose, action, 1.)
            step += 1
            reply = self.next(step, action)
            self.assertLessEqual(step,32)
        self.assertIsNone(self.agent._pending_report)
        self.assertFalse(self.node.reported)

    def test_cannot_cross_wall_or_see_through_it(self):
        grid = self.agent._frontier_grid
        wall = grid.world_to_cell((.60,0.))[0]
        grid.free[:,wall] = False
        grid.raw_obstacle[:,wall] = grid.obstacle[:,wall] = True
        self.assertIsNone(choose_view(grid,self.pose,(1.30,0.),None,1.,[]))

    def test_recognized_from_different_angle(self):
        pose = (.75,0.,-math.pi/2)
        result = choose_view(self.agent._frontier_grid,pose,(1.30,0.),None,1.,[])
        waypoint, commands = result
        self.assertIn(LEFT, commands)
        error = abs(math.remainder(math.atan2(-waypoint[1],1.30-waypoint[0])-waypoint[2],
                                   2*math.pi))
        self.assertLessEqual(error,math.radians(16))

    def test_return_to_registered_view_preserves_offcentre_yaw(self):
        start = (-.282, -.695, -1.95632)
        reference = (0.,0.,2.23246744)
        target = (-.6,0.)
        waypoint, commands = return_to_reference(self.agent._frontier_grid,start,target,
                                                  reference,1.,[],1.)
        self.assertIn(FORWARD, commands)
        self.assertLessEqual(math.dist(waypoint[:2], reference[:2]),.14)
        self.assertLessEqual(abs(math.remainder(waypoint[2]-reference[2],2*math.pi)),
                             math.radians(15))
        self.assertIsNone(return_to_reference(self.agent._frontier_grid,start,target,reference,
                                              1.,[waypoint[:2]],1.))


if __name__ == "__main__":
    unittest.main()
