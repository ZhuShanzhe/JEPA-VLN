"""Offline checks through the normal explicit / automatic report entry points."""
import unittest
from unittest.mock import Mock
import numpy as np
from benchmark_api import Action
from report_recovery import ReportRecoveryMixin
from report_evidence import eligible_boxes, current_support
from appearance import build_reference, match_reference
from test_chain import ReportChainTests as Fixtures
from trace_agent import TraceFrontier


class LivePolicy(ReportRecoveryMixin,TraceFrontier):
    pass


class ReportRecoveryChecks(unittest.TestCase):
    def setUp(self):
        f=Fixtures()
        f.setUp()
        a=f.make_agent(LivePolicy)
        self.a,self.f=a,f
        self.node=f.register(a,3,(0.,.5,.5))
        a._last_feed_info["is_keyframe"]=True
        a._category=lambda _: "painting"
        a._boxes=Mock(return_value=[dict(bbox=[80,80,240,240])])
        a._explore_follower=None
        a._autonomous_explore_action=Mock(return_value=int(Action.MOVE_FORWARD))

    def test_fresh_detection_executes_only_on_ack(self):
        a=self.a
        self.assertEqual(a._report_found(self.node.iid),int(Action.TARGET_FOUND))
        self.assertEqual(a._reported_count,0)
        a._settle_report(self.f.observation(21,int(Action.TARGET_FOUND)))
        self.assertEqual(a._reported_count,1)
        self.assertTrue(self.node.reported)
        self.assertNotEqual(a._report_found(self.node.iid),int(Action.TARGET_FOUND))

    def test_uninformative_detection_blocks_both_report_routes(self):
        a=self.a
        a._boxes.return_value=[dict(bbox=[0,0,320,320])]
        self.assertIsNone(a._auto_report(a._last_observation))
        self.assertEqual(a._report_found(self.node.iid),int(Action.MOVE_FORWARD))
        self.assertFalse(self.node.reported)
        self.assertEqual(a._report_feedback[self.node.iid]["reason"],"no_local_target_box")
        self.assertEqual(a._report_feedback[self.node.iid]["recovery"],"choose_fresh_frontier")

    def test_parent_gate_near_distance_stays_required(self):
        a=self.a
        a._geometry=lambda *args:(3.,3.,3.,0.)
        self.assertFalse(a._report_gate(self.node,a._last_observation)[0])
        a._boxes.assert_not_called()

    def test_next_report_advances_existing_frontier(self):
        a=self.a
        a._boxes.return_value=[]
        a._report_found(self.node.iid)
        a._last_observation=self.f.observation(21,int(Action.MOVE_FORWARD))
        a._explore_follower=object()
        a._explore_follow=Mock(return_value=int(Action.MOVE_FORWARD))
        self.assertEqual(a._report_found(self.node.iid),int(Action.MOVE_FORWARD))
        self.assertEqual(a._report_feedback[self.node.iid]["recovery"],"follow_fresh_frontier")
        a._autonomous_explore_action.assert_called_once()

    def test_rejected_report_returns_to_reachable_saved_view(self):
        a=self.a
        a._geometry=lambda *args:(1.0,.45,.50,0.)
        a._boxes.return_value=[]
        a._schedule_revisit=Mock(return_value={"navigating":True})
        a._close_commit=a._end_scan=a._end_route=a._cancel_delay=Mock()
        a._nav_action=Mock(return_value=(int(Action.MOVE_FORWARD),False,False))
        self.assertEqual(a._report_found(self.node.iid),int(Action.MOVE_FORWARD))
        a._schedule_revisit.assert_called_once_with(3)
        self.assertEqual(a._confirm_pending,self.node.iid)

    def test_multiple_boxes_one_location_and_ambiguity(self):
        image=self.f.rgb
        ref=build_reference(self.f.jpeg,[80,80,240,240])
        match=match_reference(ref,image)
        boxes=eligible_boxes([{"bbox":[80,80,240,240]},{"bbox":[0,0,45,45]}],image.shape)
        self.assertEqual(current_support(ref,match,image,boxes)[0],(80,80,240,240))
        self.assertIsNone(current_support(ref,match,image,boxes+[(85,85,235,235)])[0])
        self.assertFalse(eligible_boxes([{"bbox":[np.nan,0,20,20]}],image.shape))

    def test_unbounded_reference_band_cannot_certify_an_instance(self):
        image=self.f.rgb
        ref=build_reference(self.f.jpeg,[0,0,320,155])
        self.assertIsNotNone(ref)
        match=match_reference(ref,image)
        self.assertEqual(current_support(ref,match,image,[(0,0,320,155)]),
                         (None,"reference_spans_opposite_borders"))
