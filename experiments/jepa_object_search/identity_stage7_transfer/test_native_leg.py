"""Synthetic-map interface tests using MOST-Nav's own A*, follower and steering."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import test_chain  # Disables IP sockets before loading the production stack.
from agents.nav_agent import NavAgent, nav
from approach_agent import ApproachAgent
from benchmark_api import Action
from transfer_agent import ArrivalTransferAgent


class NativeLegTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_chain.ReportChainTests()
        self.fixture.setUp()
        self.a = a = self.fixture.make_agent(ArrivalTransferAgent)
        memory = a.instance_store
        NavAgent._nav_reset_state(a)
        a.instance_store = memory
        # These were stubbed in the report-only fixture. Exercise the real overrides.
        del a._clear_current_target, a._return
        a.align_R = np.eye(3)
        a.reach_m, a.replan_interval, a._last_motion_failed = .8, 30, False
        a.follower = nav.PathFollower(1, reach_m=a.reach_m)
        a.follower.x, a.follower.y = .05, .05
        self.grid = self.map(wall_open=True)
        frames = [dict(frame_id=i, pose=np.eye(4)) for i in range(5)]
        a.client.get_frame_points = Mock(return_value=frames)
        a.client.get_all_poses = Mock(return_value=(None, None))
        # Supply synthetic reconstruction and poses. The actual path search,
        # steering, follower commands, temporary blocks and arrival logic stay live.
        a._build_metric_grid = Mock(side_effect=lambda *_: (self.grid, 1.))
        a._refresh_anchor = Mock()
        a._local_scale_at = lambda points: np.ones(len(points))
        a._instance_distance_m = lambda n: math.dist(
            (a.follower.x, a.follower.y), n.point[:2])
        a._collect_views = Mock()
        a._references = Mock(return_value=[])
        # Route bookkeeping for JEPA exploratory branches absent in this fixture.
        for name in ('_end_view_watch', '_cancel_delay', '_close_commit', '_end_scan', '_end_route'):
            setattr(a, name, Mock())
        a._explore_follow = Mock(return_value=int(Action.TURN_LEFT))
        a._autonomous_explore_action = Mock(return_value=int(Action.TURN_RIGHT))
        self.node = self.fixture.register(a, 3, (4.05, .05, .5))
        self.obs = self.fixture.observation(20, None)
        self.goto = SimpleNamespace(action='GOTO_INSTANCE', target_id=str(self.node.iid),
                                    as_dict=lambda: {'action':'GOTO_INSTANCE','target_id':str(self.node.iid)})
        a.decision_loop = SimpleNamespace(_trace_record=None)

    @staticmethod
    def map(wall_open):
        blocked = np.zeros((55, 65), dtype=bool)
        # The planning grid has clearance around the wall. The physical wall
        # in the open-door fixture is column 29 and ends at row 31.
        blocked[:34 if wall_open else 55, 28:31] = True
        return nav.OccupancyGrid(.1, (-1., -2.), ~blocked, blocked)

    def test_obstacle_detour_and_arrival_handoff(self):
        a = self.a
        # An eager local policy would hijack the first step or a later nav tick
        # if the distant-target lease no longer protected either entry point.
        with patch.object(ApproachAgent, '_pursue', return_value=int(Action.LOOK_UP)) as eager:
            act = a._execute_decider_result(self.obs, self.goto, 'world_state_updated')
            self.assertEqual(a._native_leg, self.node.iid)
            self.assertIsInstance(a._native_leg, int)
            self.assertEqual(a.target_instance_id, self.node.iid)
            path = list(a.follower.path)
            self.assertGreater(max(p[1] for p in path), 1.35)
            self.assertTrue(np.allclose(a.target_point, self.node.point))
            a.client.get_frame_points.assert_called_once_with(stride=6)
            a._build_metric_grid.assert_called_once()
            for step in range(21, 51):
                self.assertIn(act, (int(Action.MOVE_FORWARD), int(Action.TURN_LEFT), int(Action.TURN_RIGHT)))
                before = (a.follower.x, a.follower.y)
                a.follower.dead_reckon(act)
                after = (a.follower.x, a.follower.y)
                cells = [self.grid.world_to_cell(p) for p in np.linspace(before, after, 40)]
                self.assertFalse(any(x == 29 and y <= 31 for x,y in cells))
                obs = self.fixture.observation(step, act)
                act, arrived, stuck = a._nav_action(obs)
                self.assertFalse(stuck)
                if arrived:
                    break
            else:
                self.fail('native follower did not signal arrival')
            self.assertLess(a._instance_distance_m(self.node), a.reach_m)
            eager.assert_not_called()
            def arrival(observation):
                self.assertIsNone(a._native_leg)
                self.assertFalse(a._routing_native_goto)
                # Supply a normal near-target decision as the VLM boundary.
                return self.goto, a._execute_decider_result(observation, self.goto, 'arrival')
            a._arrival_vlm_decision = Mock(side_effect=arrival)
            self.assertEqual(a._handle_arrival_transition(obs), int(Action.LOOK_UP))
            a._arrival_vlm_decision.assert_called_once()
            eager.assert_called_once_with(obs, self.node)
            self.assertIsNone(a._pending_report)

    def test_closed_wall_records_plan_failure_without_report(self):
        a = self.a
        self.grid = self.map(wall_open=False)
        action = a._execute_decider_result(self.obs, self.goto, 'world_state_updated')
        self.assertEqual(action, int(Action.TURN_LEFT))
        self.assertEqual(a.mode, 'explore')
        self.assertIsNone(a._native_leg)
        self.assertEqual(a._instance_plan_failures[self.node.iid]['count'], 1)
        self.assertEqual(a._plan_block_counts['no_path'], 1)
        self.assertIsNone(a._pending_report)
        self.assertFalse(self.node.reported)

    def test_reported_owner_not_routed_again(self):
        a = self.a
        a.instance_store.claim(self.node)
        action = a._execute_decider_result(self.obs, self.goto, 'world_state_updated')
        self.assertEqual(action, int(Action.TURN_RIGHT))
        self.assertIsNone(a._native_leg)
        a.client.get_frame_points.assert_not_called()
        self.assertIsNone(a._pending_report)
        a._stamp_pending_outcome.assert_any_call(
            'blocked:already_reported', 'Select another instance or a fresh frontier.')


if __name__ == '__main__':
    unittest.main(verbosity=2)
