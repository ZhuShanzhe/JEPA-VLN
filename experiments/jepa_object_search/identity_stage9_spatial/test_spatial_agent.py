"""Policy entry, cross-frame adjudication, duplicate suppression and reachable view ranking."""
from types import SimpleNamespace
from unittest import TestCase,main
from unittest.mock import Mock,patch

import numpy as np
import cv2

from agents.navigator import OccupancyGrid
from confirm_view import FORWARD,LEFT
from delayed_relook_agent import PendingView
from spatial_agent import SpatialAgent,target_relative_motion
from spatial_identity import Anchor,Surface
import test_chain
from weak_route_agent import WeakRouteAgent


class SpatialRuntimeTests(TestCase):
    def setUp(self):
        f=self.f=test_chain.ReportChainTests()
        f.setUp()
        a=self.a=f.make_agent(SpatialAgent)
        self.n=f.register(a,3,(1.,0.,.4))
        self.old=f.register(a,7,(1.05,0.,.4))
        a.instance_store.claim(self.old)
        self.ref=SimpleNamespace(roi=(.20,.20,.60,.60))
        a._views[self.n.observation_ids[0]]=(3,self.ref)
        a._confirmed_reference[self.n.iid]=(3,self.ref)
        a._collect_views=lambda:None
        self.pose=(0.,0.,0.)
        a._estimated_current_pose=lambda:self.pose
        a.align_R=np.eye(3)
        free=np.ones((80,80),bool)
        a._frontier_grid=OccupancyGrid(.1,(-4,-4),free,~free)
        a._last_motion_failed=False
        self.rgb=f.rgb
        self.box=np.array([80.,80.,240.,240.])
        self.rows=[dict(bbox=self.box,score=.95)]
        a._boxes=Mock(return_value=self.rows)
        a._category=lambda obs:'clock'
        self.go=SimpleNamespace(action='REPORT_FOUND',target_id=str(self.n.iid))
        self.snapshot={'frame_id':23}
        a._map_snapshot=Mock(return_value={23:self.snapshot})
        a._depart=Mock(return_value=LEFT)

    def test_full_request_enters_relook_even_with_previously_trusted_crop(self):
        a=self.a
        self.assertTrue(a._references(self.n))
        a._rank_probe=Mock(return_value=dict(sequence=(FORWARD,),waypoint=(.25,0.,0.),score=.9,source='map',count=2))
        self.assertEqual(a._execute_decider_result(self.f.observation(20,None),self.go,'arrival'),FORWARD)
        self.assertEqual(a._relook.iid,self.n.iid)
        self.assertIsNone(a._pending_report)
        a._last_feed_info={'frame_id':23,'is_keyframe':True}
        self.pose=(.25,0.,0.)
        obs=self.f.observation(21,FORWARD)
        a._capture_relook(obs,self.n,self.pose,1.)
        self.assertIn(23,a._native_captures)
        a._end_relook('no_mapped_identifiable_view')
        self.assertIn(23,a._waiting_for_map)
        a.target_instance_id=self.n.iid
        self.resolve(159)
        self.assertFalse(a._waiting_for_map)
        self.assertEqual(a._known_owner(self.n).iid,self.old.iid)
        self.assertEqual(a._consume_departure(self.f.observation(159,FORWARD)),LEFT)
        self.assertFalse(self.n.reported or a._pending_report)
        self.assertIsNone(a._consume_departure(self.f.observation(159,FORWARD)))
        self.assertFalse(a._report_gate(self.n,obs)[0])

    def resolve(self,step,ambiguous=False):
        a=self.a
        ctx=a._waiting_for_map[23]
        a._waiting_for_map[23]=PendingView(ctx.instance,ctx.observations,ctx.goal_index,ctx.category,
                                          23,step-10,ctx.camera_pose)
        surface=Surface(point=(1.,0.,.4),normal=(0.,0.,1.))
        query=Anchor(23,(1.,0.,.4),(0.,0.,1.),.04,surface,None,False)
        old=Anchor(7,(1.02,0.,.4),(0.,1.,1.),.04,surface,None,False)
        memo={self.old.iid:[old]}
        if ambiguous:
            memo['unlocalized-neighbor']=[]
        a.client.get_frame_image=Mock(return_value=({'found':True},self.f.jpeg))
        with patch('spatial_agent.rgb_aligned',return_value=True),patch('spatial_agent.skew_view',return_value={}), \
             patch('spatial_agent.grounded_region',return_value=(.02,self.box,np.array(self.n.point))), \
             patch('spatial_agent.make_anchor',return_value=query),patch.object(a,'_memory_anchors',return_value=(memo,{},set())):
            obs=self.f.observation(step,FORWARD)
            a.now=step
            a._last_observation=obs
            a._poll_pending_views(obs)

    def test_ambiguous_deferred_result_never_suppresses_neighbor(self):
        a,n=self.a,self.n
        signature=tuple(n.observation_ids)
        a._needs_view[n.iid]=signature
        a._waiting_for_map[23]=PendingView(n.iid,signature,None,'clock',23,15,(0.,.2,0.))
        self.resolve(25,True)
        self.assertIsNone(a._known_owner(n))
        self.assertEqual(a._report_gate(n,self.f.observation(26,FORWARD)),(False,'need_independent_spatial_confirmation'))
        self.assertTrue(a._weak_source(n))
        self.assertFalse(n.reported or a._pending_report)
        self.assertFalse(a._waiting_for_map)
        a._relook_history[(n.iid,signature)]=[(0,self.pose[:2]),(12,self.pose[:2])]
        self.assertEqual(a._execute_decider_result(self.f.observation(26,None),self.go,'arrival'),LEFT)
        self.assertIsNone(a._known_owner(n))

    def test_failed_explicit_arrival_arms_probe_when_no_old_nearby(self):
        self.old.point=(4.,0.,.4)
        self.assertIsNone(self.a._weak_source(self.n))
        with patch.object(WeakRouteAgent,'_report_gate',return_value=(False,'arrival_no_unique_correspondence')):
            self.assertFalse(self.a._report_gate(self.n,self.f.observation(20,None))[0])
        self.assertTrue(self.a._weak_source(self.n))

    def test_camera_map_detector_to_duplicate_on_two_real_arrays(self):
        # No ground-truth mask, registered instance position label or mocked
        # association is supplied to the classifier. Build the point-grid and
        # native RGB for translated cameras observing the same patch of wall.
        a=self.a
        size,stride,f=256,8,230.
        yy,xx=np.mgrid[0:size:stride,0:size:stride]+.5
        def picture(fid,cam_x):
            depth=np.full_like(xx,2.)
            points=np.dstack([(xx-size/2)*depth/f+cam_x,(yy-size/2)*depth/f,depth])
            c2w=np.eye(4)
            c2w[0,3]=cam_x
            x0=size/2-cam_x*f/2
            box=np.array([x0-24.,104.,x0+24.,152.])
            rgb=np.full((size,size,3),180,np.uint8)
            x,y,u,v=box.astype(int)
            rgb[y:v,x:u]=35
            snap=dict(pose=c2w,points=points.reshape(-1,3),rows=np.repeat(np.arange(0,size,stride),size//stride),
                      colors=rgb[::stride,::stride].reshape(-1,3),frame_id=fid)
            return snap,rgb,box
        src,old_rgb,old_box=picture(7,0.)
        snap,rgb,box=picture(23,.3)
        self.old.point=(0.,0.,2.)
        self.n.point=(.02,.01,2.)
        ref=SimpleNamespace(roi=tuple(old_box/size))
        a._confirmed_reference[self.old.iid]=(7,ref)
        a._trusted_pixels[(7,ref.roi)]=(old_rgb,old_box)
        a._map_snapshot=Mock(return_value={7:src,23:snap})
        _,data=cv2.imencode('.png',rgb)
        a.client.get_frame_image=Mock(return_value=({'found':True},data.tobytes()))
        a._native_captures[23]=(rgb,[dict(bbox=box,score=.9)])
        result,owner=a._inspect_capture(self.n,23,snap,1.)
        self.assertEqual(result,'duplicate')
        self.assertIs(owner,self.old)
        self.assertIsNone(a._pending_report)

    def test_shape_predictor_only_ranks_reachable_translations(self):
        a=self.a
        a._spatial_probe=self.n
        a._source=lambda *args:(self.rgb,self.box)
        a._verification_model=Mock(return_value=Mock(score=Mock(return_value=[.5,.5])))
        a._shape_view_model=Mock(score_views=Mock(return_value=np.array([.1,.9])))
        opts=[((0.,0.,0.),(LEFT,),'yaw_only'),((.25,.1,0.),(FORWARD,),'a'),((.25,-.1,0.),(FORWARD,LEFT),'b')]
        with patch('spatial_agent.map_views',return_value=opts), patch('spatial_agent.navigator.pose_to_yaw_2d',return_value=(0.,0.,0.)):
            picked=a._rank_probe(3,self.ref.roi,self.pose,(1.,0.),1.,[],5,True)
        self.assertEqual(picked['waypoint'],opts[-1][0])
        self.assertEqual(picked['count'],2)
        self.assertTrue(a._shape_view_model.score_views.called)
        angle,ratio=target_relative_motion((1.,0.,0.),(0.,-2.,0.),(0.,0.))
        self.assertAlmostEqual(angle,np.pi/2)
        self.assertAlmostEqual(ratio,2.)


if __name__=='__main__':
    main()
