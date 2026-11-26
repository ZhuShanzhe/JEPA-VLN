"""Decision-loop integration with controlled map, observations and identity verdicts."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from agents.navigator import OccupancyGrid
from benchmark_api import Action
from confirm_view import FORWARD, LEFT, pose_after
from test_chain import ReportChainTests as Fixture
from approach_agent import ApproachAgent
from approach_motion import plan_approach
from barrier_agent import BarrierAgent


class ApproachTests(unittest.TestCase):
    def setUp(self):
        self.f = Fixture()
        self.f.setUp()
        self.a = a = self.f.make_agent(ApproachAgent)
        self.node = self.f.register(a, 3, (2.75, 0., .5))
        self.ref = SimpleNamespace(roi=(.25,.25,.75,.75))
        a._rank_probe = Mock(return_value=None)
        a._collect_views = lambda: None
        a._references = lambda node: [(3, self.ref)]
        a._note = Mock()
        a._capture_report_view = lambda: None
        a._last_motion_failed = False
        self.pose = (0., 0., 0.)
        a._estimated_current_pose = lambda: self.pose
        a._instance_distance_m = lambda n: float(np.linalg.norm(np.array(n.point[:2])-self.pose[:2]))
        free = np.ones((80, 80), dtype=bool)
        a._frontier_grid = OccupancyGrid(.1, (-4., -4.), free, ~free)
        def gate(node, obs):
            if a._instance_distance_m(node) <= .65:
                a._cleared_reference = (node.iid, (obs.step_count, 3, self.ref))
                return True, "test_verified_identity"
            return False, "need_metric_near_view"
        a._report_gate = gate

    def obs(self, step, previous=None):
        obs = self.f.observation(step, previous)
        obs.max_steps = 300
        self.a.now = step
        self.a._last_observation = obs
        return obs

    def test_explicit_decision_and_explore_keep_target_until_report_and_receipt(self):
        a = self.a
        obs = self.obs(20)
        reply = a._execute_decider_result(obs, SimpleNamespace(action="GOTO_FRONTIER"), "arrival")
        self.assertEqual(reply, FORWARD)
        self.assertEqual(a._auto_report(obs), reply)
        count = 0
        while reply != int(Action.TARGET_FOUND):
            count += 1
            self.assertLess(count, 20)
            self.pose = pose_after(self.pose, reply, 1.)
            obs = self.obs(20 + count, reply)
            reply = a._explore_action(obs)
        self.assertEqual(count, 9)
        self.assertFalse(self.node.reported)
        a._settle_report(self.obs(21 + count, reply))
        self.assertEqual(a._reported_count, 1)
        self.assertTrue(self.node.reported)
        self.assertIsNone(a._pursue(self.a._last_observation))

    def test_conflict_ends_pursuit_without_reporting(self):
        a = self.a
        self.assertEqual(a._pursue(self.obs(20)), FORWARD)
        a._report_gate = lambda n,o:(False, "other_instance_explains_same_live_region")
        self.assertIsNone(a._pursue(self.obs(21, FORWARD)))
        self.assertIsNone(a._pending_report)
        self.assertIsNone(a._pursuit)

    def test_failed_forward_replans_and_does_not_reissue_same_edge(self):
        a = self.a
        self.assertEqual(a._pursue(self.obs(20)), FORWARD)
        a._last_motion_failed = True
        nxt = a._pursue(self.obs(21, FORWARD))
        self.assertIn(nxt, (int(Action.TURN_LEFT), int(Action.TURN_RIGHT)))
        self.assertFalse(self.node.reported)

    def test_last_action_available_for_valid_report_only(self):
        a = self.a
        self.assertIsNone(a._pursue(self.obs(299)))
        self.pose = (2.25,0.,0.)
        a._paused_sightings.clear()
        self.assertEqual(a._pursue(self.obs(299)), int(Action.TARGET_FOUND))

    def test_open_other_instance_and_finish_after_quota(self):
        a = self.a
        a._references = lambda node: []
        self.assertIsNone(a._pursue(self.obs(20)))
        a._reported_count = 2
        with patch.object(BarrierAgent,"_execute_decider_result",return_value=int(Action.FINISH)) as parent:
            self.assertEqual(a._execute_decider_result(self.obs(21), SimpleNamespace(action="FINISH"),
                                                      "finish_check"), int(Action.FINISH))
            parent.assert_called_once()

    def test_already_close_and_unresolved_gets_different_observation(self):
        a = self.a
        self.pose = (2.25, 0., 0.)
        self.assertEqual(plan_approach(a._frontier_grid,self.pose,(2.75,0.),1.,.65),())
        a._report_gate = lambda n,o: (False,"identity_fewer_than_two_shared_timestamped_anchors")
        reply = a._pursue(self.obs(20))
        self.assertIsNotNone(reply)
        self.assertNotEqual(reply,int(Action.TARGET_FOUND))
        self.assertEqual(a._pursuit.inspected, [(2.25,0.)])

    def test_trusted_camera_position_needs_matching_current_box_and_no_rival(self):
        a = self.a
        del a._report_gate
        view = (1.1,.2,0.)
        a._remembered_views = lambda n,s: iter([(3, self.ref, view)])
        a._category = lambda obs: "clock"
        a._boxes = Mock(return_value=[{"bbox":[90,80,190,180]}])
        a._support = Mock(return_value=((90,80,190,180),"same"))
        a._rival_claims = Mock(return_value=False)
        with patch.object(BarrierAgent,"_report_gate",return_value=(False,"need_metric_near_view")):
            self.assertFalse(a._report_gate(self.node,self.obs(20))[0])
            a._boxes.assert_not_called()
            self.pose = view
            self.assertTrue(a._report_gate(self.node,self.obs(21))[0])
            a._rival_claims.return_value = True
            self.assertFalse(a._report_gate(self.node,self.obs(22))[0])
            a._rival_claims.return_value = False
            a._boxes.return_value = []
            self.assertFalse(a._report_gate(self.node,self.obs(23))[0])
        a._support.reset_mock()
        with patch.object(BarrierAgent,"_report_gate",return_value=(False,"identity_different")):
            self.assertFalse(a._report_gate(self.node,self.obs(24))[0])
            a._support.assert_not_called()

    def test_unreachable_remembered_camera_still_allows_other_observation(self):
        self.a._remembered_views=lambda n,s: iter([(3,self.ref,(-7.,0.,0.))])
        reply = self.a._pursue(self.obs(20))
        self.assertEqual(reply, FORWARD)
        self.assertEqual(self.a._note.call_args.kwargs["view"],"object_standoff")

    def test_ranked_local_view_is_retained_but_rechecked_during_motion(self):
        self.pose = (1.25,0.,0.)
        a = self.a
        a._rank_probe.return_value = {"sequence":(FORWARD,FORWARD,LEFT)}
        self.assertEqual(a._pursue(self.obs(20)),FORWARD)
        self.assertEqual(a._note.call_args.kwargs["view"],"learned_verification_view")
        self.assertEqual(a._pursuit.actions,(FORWARD,LEFT))

    def test_single_witness_at_route_end_obtains_new_view_without_reporting(self):
        a = self.a
        self.pose = (2.25,0.,0.)
        a._report_gate = lambda n,o:(False,"identity_missing_geometry")
        a._visual_history[(3,self.ref.roi)] = (20,None,(30,80,100,165),True,False)
        self.assertEqual(a._pursue(self.obs(20)),LEFT)
        self.assertIsNone(a._pending_report)


if __name__ == "__main__":
    unittest.main()
