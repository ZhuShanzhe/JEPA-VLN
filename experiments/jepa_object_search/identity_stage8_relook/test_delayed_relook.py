"""Replay observed mapper timing; separately test identity/registration with controlled inputs."""
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import Mock, patch
import json
import numpy as np
from test_chain import ReportChainTests
from delayed_relook_agent import DelayedRelookAgent, PendingView
from weak_route_agent import WeakRouteAgent, Relook
from trusted_reference import ReferenceEvidence


class DeferredTests(TestCase):
    def setUp(self):
        self.f=ReportChainTests()
        self.f.setUp()
        self.a=a=self.f.make_agent(DelayedRelookAgent)
        self.node=n=self.f.register(a,3,(1.,0.,.4))
        a.align_R=np.eye(3)
        a.now=149
        a._category=lambda obs:'clock'
        a._references=lambda node: []
        a._collect_views=lambda:None
        a._note=Mock()
        a._map_snapshot=Mock(return_value={})
        self.pose=(.25,0.,0.)
        self.rgb=np.random.default_rng(13).integers(0,256,(320,320,3),np.uint8)

    def enqueue(self,fid,t):
        a=self.a
        n=self.node
        a._relook=Relook(n.iid,t-5,self.pose,tuple(n.point[:2]),(1,))
        a._boxes=Mock(return_value=[dict(bbox=[84,84,180,180],score=.95)])
        a._last_feed_info={'is_keyframe':True,'frame_id':fid}
        a._relook.pose=(0.,0.,0.)
        a._capture_relook(self.f.observation(t,1),n,self.pose,1.)
        a._end_relook('no_mapped_identifiable_view')
        self.assertEqual(a._waiting_for_map[fid].camera_pose,self.pose)
        self.assertEqual(a._waiting_for_map[fid].observations,tuple(n.observation_ids))
        self.assertIsNone(a._relook)

    def poll(self,t):
        a=self.a
        a.now=t
        obs=self.f.observation(t,1)
        a._last_observation=obs
        a._poll_pending_views(obs)
        return obs

    def test_archived_timing(self):
        file=Path(__file__).with_name('relook_capture_audit.json')
        cases=json.loads(file.read_bytes())['captures']
        for case in cases:
            fid=case['frame_id']
            self.enqueue(fid,case['capture_step'])
            checked=case['processed_by_step']
            self.a._map_snapshot=Mock(side_effect=lambda:{fid:{'frame_id':fid}} if self.a.now>=checked else {})
            with patch.object(WeakRouteAgent,'_inspect_capture',return_value=('no_unique_spatial_box',None)) as check, \
                    patch.object(WeakRouteAgent,'_record_and_update') as native_finish:
                def motion_tick(step):
                    obs=self.f.observation(step,1)
                    self.a.now=step
                    self.a._last_observation=obs
                    self.a._record_and_update(obs,1)
                    native_finish.assert_called_with(obs,1)
                for t in range(case['end_step'],checked):
                    motion_tick(t)
                    check.assert_not_called()
                    self.assertIn(fid,self.a._waiting_for_map)
                motion_tick(checked)
                check.assert_called_once_with(self.node,fid,{'frame_id':fid},1.)
                motion_tick(checked+1)
                self.assertEqual(check.call_count,1)
                self.assertFalse(self.a._pending_report)
                self.assertFalse(self.node.reported)
                self.assertFalse(self.a._waiting_for_map)

    def test_old_frame_registers_only_using_its_own_pixels_and_geometry(self):
        a=self.a
        n=self.node
        self.enqueue(150,149)
        a._map_snapshot.return_value={150:{'frame_id':150}}
        a._last_feed_info={'is_keyframe':True,'frame_id':162}
        a.client.get_frame_image=Mock(return_value=({'found':True},b'old_jpeg'))
        with patch('weak_route_agent._rgb',return_value=self.rgb) as decode, \
                patch('weak_route_agent.rgb_aligned',return_value=True),patch('weak_route_agent.skew_view',return_value={}), \
                patch('weak_route_agent.grounded_region',return_value=(.05,np.array([84,84,180,180]),np.array([1.,0.,.4]))), \
                patch('weak_route_agent.assess_reference',return_value=ReferenceEvidence('supported',.7,.95)):
            obs=self.poll(159)
        a.client.get_frame_image.assert_called_once_with(150)
        decode.assert_called_once_with(b'old_jpeg')
        rec=a.instance_store.get_observation(n.observation_ids[-1])
        self.assertEqual((rec.step,rec.frame_id),(149,150))
        self.assertEqual(len(n.observation_ids),2)
        self.assertFalse(a._report_gate(n,obs)[0])
        self.assertFalse(n.reported or a._pending_report)

    def test_changed_signature_discards_without_inspection(self):
        self.enqueue(150,149)
        n=self.node
        n.observation_ids.append(9999)
        self.poll(159)
        self.a._map_snapshot.assert_not_called()
        self.assertFalse(self.a._waiting_for_map)

    def test_duplicate_does_not_interrupt_new_target(self):
        a=self.a
        self.enqueue(150,149)
        other=self.f.register(a,4,(2.,.2,.5))
        a.target_instance_id=other.iid
        a._map_snapshot.return_value={150:{}}
        a._depart=Mock(return_value=2)
        with patch.object(WeakRouteAgent,'_inspect_capture',return_value=('duplicate',other)):
            obs=self.poll(159)
        self.assertIsNone(a._consume_departure(obs))
        a._depart.assert_not_called()
        self.assertTrue(a._cooling(self.node,159))

    def test_duplicate_redirects_only_current_pursuit_once(self):
        a=self.a
        self.enqueue(150,149)
        a.target_instance_id=self.node.iid
        a._map_snapshot.return_value={150:{}}
        a._depart=Mock(return_value=2)
        old=self.f.register(a,4,(1.,0.,.42))
        a.instance_store.claim(old)
        with patch.object(WeakRouteAgent,'_inspect_capture',return_value=('duplicate',old)):
            obs=self.poll(159)
        self.assertEqual(a._consume_departure(obs),2)
        self.assertIsNone(a._consume_departure(obs))
        a._depart.assert_called_once_with(obs,self.node)


if __name__=='__main__':
    main()
