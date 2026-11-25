import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from regions import heading_change_degrees
from reid_agent import ReIDFrontier
from report_agent import ReportFrontier
from test_chain import ReportChainTests as _Fixtures


class AdapterChecks(unittest.TestCase):
    def setUp(self):
        fixture = _Fixtures("test_canonical_memory_changes_after_observed_report_only")
        fixture.setUp()
        self.agent = agent = fixture.make_agent(ReIDFrontier)
        self.node = fixture.register(agent, 3, (0, .5, .5))
        self.current = agent.instance_store.new_observation(
            (.05, .5, .5), "painting", frame_id=7, step=19,
            bbox=(80,80,240,240), candidate_id="proposal-seven")
        agent.instance_dup_radius_m = 1.0
        agent._tool_view_instance = Mock(return_value=b"existing")
        agent.client.get_candidate_evidence = Mock(return_value=({"found":True}, b"new"))
        agent.client.get_frame_pose = lambda fid: self.pose(0 if fid == 3 else 45)
        agent._matcher = Mock(return_value=Mock(compare=Mock(
            return_value={"score":.94,"threshold":.9,"match":True})))
        self.native_row = {"observation_id":self.current.oid, "frame_id":7, "candidate_id":"proposal-seven",
                           "neighbors":[{"instance_id":self.node.iid,"dist_m":.05,"reported":False}]}
        self.fixture = fixture

    @staticmethod
    def pose(heading_deg):
        a = math.radians(heading_deg)
        p = np.eye(4)
        p[:3,:3] = np.array([[-math.sin(a), 0, math.cos(a)], [math.cos(a), 0, math.sin(a)], [0, 1, 0]])
        return p

    def payload(self):
        self.agent._last_dup_reviews = [self.native_row]
        with patch.dict(os.environ, {"NAV_REID_WEIGHTS":"mock"}):
            rows, images, omitted = self.agent._dup_review_payload()
        self.assertEqual(omitted,0)
        self.assertEqual(len(images),2)
        return rows[0]

    def test_native_duplicate_tool_and_executed_only_return_guard(self):
        a = self.agent
        a.now = 12
        issued = a._stage_report(self.node, self.fixture.observation(12,None), "synthetic_close_view")
        a._settle_report(self.fixture.observation(13,issued))
        a.now = 20
        hint = self.payload()["cross_view_evidence"]
        self.assertTrue(hint["report_executed"])
        self.assertEqual(hint["view_change_deg"], -45)
        self.assertIsNone(a.instance_store.instance_for_observation(self.current.oid))
        a._tool_resolve_duplicate(self.current.oid, "DUPLICATE", self.node.iid)
        self.assertIs(a.instance_store.instance_for_observation(self.current.oid), self.node)
        a._autonomous_explore_action = Mock(return_value=2)
        a._execute_decider_result(a._last_observation, SimpleNamespace(action="GOTO_INSTANCE",
                                  target_id=self.node.iid), "synthetic")
        a._autonomous_explore_action.assert_called_once()
        self.assertEqual(len(a.instance_store.report_claims),1)

    def test_reliable_first_registration_remains_searchable(self):
        self.assertFalse(self.payload()["cross_view_evidence"]["report_executed"])
        self.assertIn(self.node,self.agent.instance_store.available())
        self.assertIsNone(self.agent.instance_store.instance_for_observation(self.current.oid))
        out = self.agent._tool_resolve_duplicate(self.current.oid,"NEW")
        self.assertNotEqual(out["instance_id"],self.node.iid)
        self.assertEqual(len(self.agent.instance_store.available()),2)

    def test_competing_instance_abstains_before_model(self):
        self.fixture.register(self.agent, 8, (.07, .5, .5))
        self.assertNotIn("cross_view_evidence",self.payload())
        self.agent._matcher.assert_not_called()

    def test_first_incoming_region_is_saved_before_future_review(self):
        a=self.agent
        def ingest(*args):
            return self.fixture.register(a, 55, (1.,1.,1.))
        with patch.object(ReportFrontier,"_ingest_semantic_hits",side_effect=ingest), \
             patch.dict(os.environ, {"NAV_REID_WEIGHTS":"mock"}):
            first=a._ingest_semantic_hits(a._last_observation,[],False)
        rec=a._records(first)[0]
        self.assertIn(rec.oid,a._region_cache)
        a.client.get_frame_image=Mock(side_effect=AssertionError("mapper image expired"))
        self.assertIs(a._crop(rec),a._region_cache[rec.oid])
        self.assertIn(first,a.instance_store.available())

    def test_height_and_goal_protect_separate_objects(self):
        self.current.point[2] += 1.
        self.assertNotIn("cross_view_evidence",self.payload())
        self.current.point[2] -= 1.
        self.current.evidence["goal_index"]=2
        self.agent._instance_goal_index[self.node.iid]=1
        self.assertNotIn("cross_view_evidence",self.payload())
        self.assertIn("error", self.agent._tool_resolve_duplicate(self.current.oid,"DUPLICATE",self.node.iid))
        self.agent._matcher.assert_not_called()

    def test_same_frame_disjoint_boxes_protect_new_object(self):
        self.current.frame_id = 3
        self.current.bbox = (0,0,30,30)
        self.assertNotIn("cross_view_evidence",self.payload())
        self.assertIn("error",self.agent._tool_resolve_duplicate(self.current.oid,"DUPLICATE",self.node.iid))

    def test_uncertain_score_preserves_new(self):
        self.agent._matcher.return_value.compare.return_value={"score":.1, "threshold":.9, "match":False}
        self.assertNotIn("cross_view_evidence",self.payload())
        out=self.agent._tool_resolve_duplicate(self.current.oid,"NEW")
        self.assertNotEqual(out["instance_id"], self.node.iid)

    def test_missing_pose_or_scale_abstains(self):
        self.agent._metric_scale_value=lambda: None
        self.assertNotIn("cross_view_evidence",self.payload())
        self.agent._metric_scale_value=lambda:1.
        self.agent.client.get_frame_pose=lambda i:None
        self.assertNotIn("cross_view_evidence",self.payload())
        self.agent._matcher.assert_not_called()

    def test_disabled_keeps_native_payload(self):
        self.agent._last_dup_reviews = [self.native_row]
        with patch.dict(os.environ, {"NAV_REID_WEIGHTS":""}):
            rows,images,omitted=self.agent._dup_review_payload()
        self.assertNotIn("cross_view_evidence",rows[0])
        self.agent._matcher.assert_not_called()

    def test_heading_wrap_and_invalid_pose(self):
        self.assertAlmostEqual(heading_change_degrees(self.pose(179),self.pose(-179),np.eye(3)),-2)
        self.assertIsNone(heading_change_degrees(np.eye(4),self.pose(0),np.eye(3)))
