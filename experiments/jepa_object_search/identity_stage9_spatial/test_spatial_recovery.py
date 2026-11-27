"""Off-track recognition must never equate an observed owner and a separate pursued clue."""
from dataclasses import replace
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from spatial_identity import Anchor, Surface
from spatial_agent import _SpatialCapture
from spatial_recovery import Decision, OwnerView, ReferenceView, RecoverySpatialAgent, recognize_seen


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        plane=Surface((0.,0.,2.),(0.,0.,-1.))
        self.ref=Anchor(21,(0.,0.,2.),(0.,0.,0.),.045,plane)
        self.query=replace(self.ref,frame_id=37,camera=(.3,0.,0.),center=(.035,0.,2.))
        self.track=(1.,0.,2.)
        self.owner=OwnerView(self.ref.center,True,(ReferenceView(self.ref,False,False),))

    def run_case(self,old=None,query=None,**kwargs):
        return recognize_seen(query or self.query,self.track,{1:old or self.owner,**kwargs})

    def test_supported_old_place_is_a_sighting_not_pursued_identity(self):
        v=self.run_case()
        self.assertEqual((v.status,v.owner),('same',1))
        self.assertEqual(recognize_seen(self.query,self.query.center,{1:self.owner}).status,'uncertain')

    def test_joint_visual_fills_absent_plane_with_metric_evidence(self):
        q=replace(self.query,surface=None)
        joint=replace(self.owner,references=(ReferenceView(self.ref,True,True),))
        body_only=replace(self.owner,references=(ReferenceView(self.ref,True,False),))
        self.assertEqual(self.run_case(joint,q).reason,'observed_old_joint_visual_place')
        self.assertEqual(self.run_case(body_only,q).status,'uncertain')

    def test_appearance_cannot_override_different_position_surface_or_intrinsic_color(self):
        red=replace(self.ref,hue=10.)
        joint=replace(self.owner,references=(ReferenceView(red,True,True),))
        for q in [replace(self.query,center=(.45,0.,2.)),replace(self.query,surface=None,hue=230.),
                  replace(self.query,surface=Surface((0.,0.,2.),(0.,-1.,0.))),
                  replace(self.query,frame_id=red.frame_id)]:
            self.assertEqual(self.run_case(joint,q).status,'uncertain')

    def test_missing_owner_reference_and_competing_unreported_neighbor_block_match(self):
        unknown=replace(self.owner,references=(),reported=False)
        self.assertEqual(self.run_case(unknown).status,'uncertain')
        self.assertEqual(self.run_case(other=unknown).status,'uncertain')
        self.assertEqual(self.run_case(other=replace(unknown,center=(-.9,0.,2.))).status,'same')

    def test_repeated_frame_cannot_create_alias_cooldown_or_replace_reference(self):
        a=object.__new__(RecoverySpatialAgent)
        n=SimpleNamespace(iid=3,observation_ids=[17])
        a.instance_store=SimpleNamespace(get=lambda iid:SimpleNamespace(iid=iid,reported=True))
        a._old_sightings={}
        a._off_track_views={}
        a._needs_view={}
        a._note=Mock()
        v=Decision('same',1,'observed_old_mapped_place')
        self.assertTrue(a._remember_offtrack(n,37,v))
        self.assertFalse(a._remember_offtrack(n,37,v))
        self.assertEqual(a._old_sightings,{1:{37:v.reason}})
        self.assertEqual(a._needs_view,{3:(17,)})
        self.assertFalse(hasattr(a,'_spatial_alias'))
        self.assertFalse(hasattr(a,'_duplicate_cooldown'))
        a._note.assert_called_once()

    def test_historical_capture_enters_fallback_and_keeps_real_node_unbound(self):
        a=object.__new__(RecoverySpatialAgent)
        n=SimpleNamespace(iid=3,point=self.track,observation_ids=[17],reported=False)
        rgb=np.full((20,20,3),90,np.uint8)
        a.align_R=np.eye(3); a.client=SimpleNamespace(get_frame_image=lambda fid:({'found':True},b'test'))
        a._native_captures={37:(rgb,[{'score':.92,'bbox':[2,2,16,16]}])}
        a._map_snapshot=Mock(return_value={37:{}})
        a._old_sightings={}; a._off_track_views={}; a._needs_view={}
        a._note=Mock(); a.instance_store=SimpleNamespace(get=lambda iid:n if iid==3 else SimpleNamespace(iid=iid,reported=True))
        a._recovery_owners=Mock(return_value={1:self.owner})
        a._last_pending_check=-1; a._relook=None; a._known_owner=lambda node:None
        a._instance_goal_index={}; a._reference_purpose=lambda flag:nullcontext(); a._pending_report=None
        a._metric_scale_value=lambda:1.; a._category=lambda obs:'clock'; a._references=lambda node:[]
        a._waiting_for_map={37:SimpleNamespace(instance=3,observations=(17,),goal_index=None,
                category='clock',captured_step=41)}
        a._duplicate_cooldown={}; a._spatial_alias={}
        with (patch.object(_SpatialCapture,'_inspect_capture',return_value=('no_unique_spatial_box',None)),
              patch('spatial_recovery._rgb',return_value=rgb),
              patch('spatial_recovery.rgb_aligned',return_value=True),
              patch('spatial_recovery.skew_view',return_value={}),
              patch('spatial_recovery.make_anchor',return_value=self.query)):
            a._poll_pending_views(SimpleNamespace(step_count=50))
            self.assertEqual(a._needs_view,{3:(17,)})
            self.assertEqual(a._waiting_for_map,{})
            self.assertFalse(a._duplicate_cooldown or a._spatial_alias)
            self.assertEqual(a._old_sightings,{1:{37:'observed_old_mapped_place'}})


if __name__=='__main__':
    unittest.main()
