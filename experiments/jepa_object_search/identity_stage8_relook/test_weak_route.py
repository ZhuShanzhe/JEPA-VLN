"""Regressions at the planner, source-registration, and region-switch interfaces."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
from agents.navigator import OccupancyGrid
from confirm_view import FORWARD,LEFT
from test_chain import ReportChainTests as Fixture
from trusted_reference import ReferenceEvidence
from weak_route_agent import WeakRouteAgent, grounded_region


class WeakRouteTests(unittest.TestCase):
    def setUp(self):
        self.fixture=Fixture()
        self.fixture.setUp()
        self.a=a=self.fixture.make_agent(WeakRouteAgent)
        self.n=self.fixture.register(a,3,(1.,0.,.4))
        self.ref=SimpleNamespace(roi=(.2,.2,.6,.6))
        a._views[self.n.observation_ids[0]]=(3,self.ref)
        a._references=lambda n: []
        a._collect_views=lambda:None
        a._note=Mock()
        a.align_R=np.eye(3)
        free=np.ones((80,80),bool)
        a._frontier_grid=OccupancyGrid(.1,(-4,-4),free,~free)
        self.pose=(0.,0.,0.)
        a._estimated_current_pose=lambda:self.pose
        a._metric_scale_value=lambda:1.
        a._rank_probe=Mock(return_value=dict(sequence=(FORWARD,),waypoint=(.25,0.,0.),score=.8,
                                             source='map_probe',count=3))
        a._boxes=Mock(return_value=[dict(bbox=[80,80,180,180],score=.9)])
        a._capture_report_view=lambda:None
        a._category=lambda o:'clock'
        a._map_snapshot=Mock(return_value={})
        a._plan_exploration=Mock()
        a._active_frontier_key=None
        a._last_frontier_clusters=[dict(world=(1.,0.)),dict(world=(4.,0.))]
        a._activate_frontier=Mock(return_value=True)
        a._explore_follow=Mock(return_value=LEFT)
        a._last_motion_failed=False
        self.obs=lambda t,prev:self.fixture.observation(t,prev)
        self.go=SimpleNamespace(action='REPORT_FOUND',target_id=str(self.n.iid))
        self.rgb=np.random.default_rng(3).integers(0,256,(260,320,3),np.uint8)

    def trigger(self):
        obs=self.obs(20,None)
        self.a.now=20
        action=self.a._execute_decider_result(obs,self.go,'arrival')
        self.assertEqual(action,FORWARD)
        self.assertTrue(self.a._rank_probe.call_args.kwargs['uncertain'])
        self.assertEqual(self.a._try_relook(obs),FORWARD)
        self.assertFalse(self.n.reported)

    def test_new_capture_registers_only_after_motion_and_verified_geometry(self):
        a=self.a
        self.trigger()
        a._last_feed_info={'frame_id':9,'is_keyframe':True}
        self.pose=(.25,0.,0.)
        obs=self.obs(21,FORWARD)
        a.now=21
        a._last_observation=obs
        a.client.get_frame_image=Mock(return_value=({'found':True},b'jpeg'))
        a._map_snapshot=Mock(return_value={9:{}})
        with patch('weak_route_agent._rgb',return_value=self.rgb),patch('weak_route_agent.rgb_aligned',return_value=True), \
             patch('weak_route_agent.skew_view',return_value={}),patch('weak_route_agent.grounded_region',
             return_value=(.05,np.array([80,80,180,180]),np.array([1.,0.,.4]))),patch(
             'weak_route_agent.assess_reference',return_value=ReferenceEvidence('supported',.6,.9)):
            self.assertIsNone(a._try_relook(obs))
        self.assertEqual(len(self.n.observation_ids),2)
        self.assertFalse(self.n.reported)
        oid=self.n.observation_ids[-1]
        self.assertEqual(a._views[oid][0],9)
        self.assertTrue(a._reference_evidence[oid].eligible())
        self.assertFalse(a._report_gate(self.n,obs)[0])

    def test_duplicate_moves_to_separate_frontier_without_claiming_other_record(self):
        a=self.a
        old=self.fixture.register(a,5,(1.,0.,.42))
        a.instance_store.claim(old)
        self.trigger()
        self.pose=(.25,0.,0.)
        a._last_feed_info={'frame_id':9,'is_keyframe':True}
        a._map_snapshot=Mock(return_value={9:{}})
        obs=self.obs(21,FORWARD)
        a.now=21
        a._last_observation=obs
        a.client.get_frame_image=Mock(return_value=({'found':True},b'jpeg'))
        a._arrival_matches=Mock(return_value={0:(5,self.ref)})
        with patch('weak_route_agent._rgb',return_value=self.rgb),patch('weak_route_agent.rgb_aligned',return_value=True), \
             patch('weak_route_agent.skew_view',return_value={}),patch('weak_route_agent.grounded_region',
             return_value=(.05,np.array([80,80,180,180]),np.array([1.,0.,.4]))),patch(
             'weak_route_agent.assess_reference',return_value=ReferenceEvidence('supported',.6,.9)):
            self.assertEqual(a._try_relook(obs),LEFT)
        self.assertFalse(self.n.reported)
        self.assertTrue(a._cooling(self.n,21))
        self.assertEqual(a._activate_frontier.call_args.args[1]['world'],(4.,0.))
        self.assertFalse(a._pending_report)

    def test_cancel_after_interrupted_motion(self):
        self.trigger()
        self.assertIsNone(self.a._try_relook(self.obs(21,LEFT)))
        self.assertIsNone(self.a._relook)
        self.assertEqual(len(self.n.observation_ids),1)
        self.a._map_snapshot.assert_not_called()

    def test_two_candidate_boxes_do_not_establish_unique_grounding(self):
        view={'grid':np.zeros((32,40,3)),'stride':8}
        rows=[{'bbox':[2,2,25,24],'score':.9},{'bbox':[31,3,50,32],'score':.85}]
        with patch('weak_route_agent.point_in_box',side_effect=[[1,0,.4],[1.,.05,.4]]):
            self.assertIsNone(grounded_region(view,self.rgb,rows,self.n.point,np.eye(3),1.))
        with patch('weak_route_agent.point_in_box',return_value=[4,0,.4]):
            self.assertIsNone(grounded_region(view,self.rgb,rows[:1],self.n.point,np.eye(3),1.))


if __name__=='__main__':
    unittest.main()
