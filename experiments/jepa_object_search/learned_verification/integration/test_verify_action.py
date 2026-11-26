"""Native review tool, map actions and fresh-frame handoff with fixed sensor replies."""
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from agents.navigator import OccupancyGrid
from benchmark_api import Action
from candidate_scout import CandidateScoutMixin
from confirm_view import pose_after, FORWARD
from learned_map_views import map_views, relative_action
from run_agent import LearnedVerificationFrontier
import test_chain


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_chain.ReportChainTests()
        self.fixture.setUp()
        self.a=a=self.fixture.make_agent(LearnedVerificationFrontier)
        folder=tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        a.output_dir=folder.name
        a._queued_pick_results=[]
        a._pending_decision_images=[]
        a._dup_review_payload=lambda:([],[],0)
        a._record_rejected_spot=Mock()
        a._last_motion_failed=False
        self.pose=(-.35,0.,0.)
        a._estimated_current_pose=lambda:self.pose
        free=np.ones((80,80),bool)
        a._frontier_grid=OccupancyGrid(.08,(-3.2,-3.2),free,~free)
        a._proposal_queue["p1"]=dict(candidate_id="p1",frame_id=3,bbox=[80,85,180,185],
                pixel_norm=[400,430],status="pending",query="clock",step=20,goal_index=None)
        a.client.resolve_candidates=Mock(return_value={"candidates":{"p1":{"found":True,"point":[.85,0,.3]}}})
        a.client.capture_keyframe=Mock(return_value={"frame_id":21})
        a._last_feed_info={"is_keyframe":False,"frame_id":2}

    def review(self,verdict):
        return self.a._tool_commit_candidates([dict(candidate_id="p1",verdict=verdict)],"clock")

    def tick(self,step,previous):
        obs=self.fixture.observation(step,previous)
        self.a._last_observation=obs
        self.a.now=step
        return self.a._auto_report(obs)

    def test_uncertain_acquires_new_view_without_registering(self):
        self.assertEqual(self.review("UNCERTAIN")["uncertain"],["p1"])
        act=self.tick(21,None)
        n=0
        while act is not None:
            self.assertIn(act,(int(Action.MOVE_FORWARD),int(Action.TURN_LEFT),int(Action.TURN_RIGHT)))
            self.pose=pose_after(self.pose,act,1.)
            n+=1
            self.assertLessEqual(n,12)
            act=self.tick(21+n,act)
        a=self.a
        self.assertGreater(n,0)
        self.assertFalse(a.instance_store.available() or a._pending_report)
        self.assertEqual(a._proposal_queue["p1"]["status"],"uncertain")
        self.assertEqual(a._recent_inspection["frame_id"],21)
        self.assertEqual(a.client.capture_keyframe.call_count,1)
        a._ev_deciding=True
        with patch.object(CandidateScoutMixin,"_build_decider_input",side_effect=lambda *args:({},None)):
            state,_=a._build_decider_input(a._last_observation)
        self.assertEqual(state["verification_observation"]["new_frame_id"],21)
        self.assertEqual(a._pending_decision_images[0][0],"inspection_p1")

    def test_missing_depth_preserves_review_state(self):
        self.review("UNCERTAIN")
        self.a.client.resolve_candidates.return_value={"candidates":{}}
        self.assertIsNone(self.tick(21,None))
        self.assertIsNone(self.a._weak_candidate)
        self.assertFalse(self.a.instance_store.available())

    def test_pending_nearby_hypotheses_keep_their_own_evidence(self):
        a=self.a
        a._proposal_queue["p2"]=dict(a._proposal_queue["p1"],candidate_id="p2",frame_id=4)
        reviews=[dict(candidate_id=k,verdict="UNCERTAIN") for k in ("p1","p2")]
        a._tool_commit_candidates(reviews,"clock")
        self.assertEqual(a._weak_candidate["cid"],"p1")
        self.assertEqual([q["cid"] for q in a._weak_waiting],["p2"])
        a.client.resolve_candidates.return_value={"candidates":{}}
        self.tick(21,None)
        a._next_weak()
        self.assertEqual((a._weak_candidate["cid"],a._weak_candidate["fid"]),("p2",4))
        self.assertFalse(a.instance_store.available())

    def test_upper_uncertainty_loads_separate_context_head(self):
        with patch.dict(os.environ, JEPA_UNCERTAIN_HEAD="context"):
            self.assertEqual(type(self.a._verification_model(True)).__name__,"WeakViewScorer")
            self.assertEqual(type(self.a._verification_model(False)).__name__,"ActiveViewScorer")

    def test_rejected_or_unknown_is_not_pursued(self):
        self.review("REJECT")
        self.assertIsNone(self.tick(21,None))
        self.assertIsNone(self.a._weak_candidate)
        out=self.a._tool_commit_candidates([dict(candidate_id="1",verdict="UNCERTAIN")])
        self.assertIn("error",out)
        self.assertIsNone(self.a._weak_candidate)

    def test_interrupted_motion_does_not_claim_a_view(self):
        self.review("UNCERTAIN")
        action=self.tick(21,None)
        self.assertIsNotNone(action)
        self.assertIsNone(self.tick(22,99))
        self.assertIsNone(self.a._recent_inspection)
        self.assertFalse(self.a.instance_store.available())

    def test_uncertain_stall_switches_direction_without_registering(self):
        self.review("UNCERTAIN")
        a=self.a
        # Two candidate routes from the same observed source and map. The second
        # selection follows a failed real action receipt and gets one less step.
        first=dict(sequence=(FORWARD,FORWARD),count=2,score=.6)
        turn=int(Action.TURN_LEFT)
        second=dict(sequence=(turn,),count=1,score=.5)
        with patch.object(a,"_rank_probe",side_effect=[first,second]) as rank:
            self.assertEqual(self.tick(21,None),FORWARD)
            a._last_motion_failed=True
            self.assertEqual(self.tick(22,FORWARD),turn)
            self.assertEqual(rank.call_args.args[-1],11)
            self.assertTrue(a._blocked_probe_edges)
            self.pose=pose_after(self.pose,turn,1.)
            a._last_motion_failed=False
            self.assertIsNone(self.tick(23,turn))
        self.assertIsNotNone(a._recent_inspection)
        self.assertFalse(a.instance_store.available() or a._pending_report)

    def test_uncertain_stops_for_review_when_fresh_region_is_clearer(self):
        self.review("UNCERTAIN")
        route=dict(sequence=(FORWARD,FORWARD,FORWARD),count=2,score=.7)
        with patch.object(self.a,"_rank_probe",return_value=route), patch.object(self.a,"_clearer_weak_view",return_value=True):
            self.assertEqual(self.tick(21,None),FORWARD)
            self.pose=pose_after(self.pose,FORWARD,1.)
            self.assertIsNone(self.tick(22,FORWARD))
        self.assertEqual(self.a._recent_inspection["step"],22)
        self.assertFalse(self.a.instance_store.available() or self.a._pending_report)

    def test_agent_left_heading_converts_to_camera_right_yaw(self):
        np.testing.assert_allclose(relative_action((0,0,0),(1,-.4,-.2),1.),(1,.4,.2))
        np.testing.assert_allclose(relative_action((0,0,np.pi/2),(.4,1,np.pi/2-.2),1.),(1,.4,.2))

    def test_closed_wall_has_no_reachable_view(self):
        grid=self.a._frontier_grid
        wall=grid.world_to_cell((.2,0))[0]
        grid.free[:,wall]=False
        grid.raw_obstacle[:,wall]=grid.obstacle[:,wall]=True
        opts=map_views(grid,(-.35,0,0),(.85,0.),(.9,.05,0.),1.,[],.8)
        self.assertEqual(opts,[])

    def test_failed_edge_excluded_for_remaining_budget(self):
        p=self.pose
        opts=map_views(self.a._frontier_grid,p,(.85,0.),(0.,0.,0.),1.,[],.8,[p],6)
        self.assertTrue(opts)
        self.assertTrue(all(route[0]!=FORWARD and len(route)<=6 for _,route,_ in opts))

    def test_true_stall_requests_one_bounded_alternative(self):
        a=self.a
        node=self.fixture.register(a,3,(.85,0.,.25))
        a._collect_views()
        step=22
        a._view_history[node.iid]=dict(attempts=1,end=-100,excluded=[])
        a._confirm_view_state=dict(node_id=node.iid,waypoint=(.15,0.,0.),fid=3,target=(.85,0.),
            start=20,sequence=(FORWARD,FORWARD),cursor=1,last_step=21,expected=(-.10,0.,0.),learned=True,budget=7)
        a._last_motion_failed=True
        obs=self.fixture.observation(step,FORWARD);a.now=step;a._last_observation=obs
        reply=a._confirm_view_step(node,obs)
        self.assertTrue(a._blocked_probe_edges)
        self.assertIsNotNone(reply)
        self.assertNotEqual(reply[0],FORWARD)
        self.assertLessEqual(len(a._confirm_view_state["sequence"]),6)
        self.assertEqual(a._view_history[node.iid]["attempts"],2)

if __name__=="__main__":
    unittest.main()
