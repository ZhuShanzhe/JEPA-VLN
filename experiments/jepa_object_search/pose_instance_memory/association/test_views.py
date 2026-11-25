import os
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np

from geometry_links import match_views, surface_point
from map_views import metric_view, point_in_box
from multiview_agent import MultiviewFrontier, ReIDFrontier
import test_chain


def snapshot(fid):
    stride=8
    u,v=np.meshgrid(np.arange(40)*stride+.5,np.arange(30)*stride+.5)
    pose=np.eye(4)
    pose[0,3]=0 if fid==1 else .12
    pts=np.stack(((u-160)/200*2+pose[0,3],(v-120)/200*2,np.full_like(u,2.)),axis=-1)
    return dict(frame_id=fid,points=pts.reshape(-1,3),rows=(v-.5).astype(int).ravel(),
                pose=pose,colors=np.zeros((len(u.ravel()),3),np.uint8))


def view(fid,boxes=None):
    v=metric_view(snapshot(fid),1.,8)
    shift=0 if fid==1 else -12
    if boxes is None:
        boxes=[[82+shift,80,125+shift,150],[139+shift,80,184+shift,150]]
    return dict(v,boxes=boxes,points=[point_in_box(v,b) for b in boxes])


class GeometryChecks(unittest.TestCase):
    def test_view_motion_preserves_both_and_separates_crossed_pair(self):
        g=match_views(view(1),view(2))
        self.assertEqual(g["matches"],[(0,0),(1,1)])
        self.assertEqual(set(g["distinct"]),{(0,1),(1,0)})

    def test_repeat_rectangles_make_geometric_identity_ambiguous(self):
        v=view(1)
        w=view(2,[v["boxes"][0],v["boxes"][0]])
        g=match_views(v,w)
        self.assertEqual(g["matches"],[])
        self.assertEqual(g["same_frame_distinct_right"],[])

    def test_occluded_partner_abstains_from_negative(self):
        g=match_views(view(1),view(2,[[70,80,113,150]]))
        self.assertEqual(g["matches"],[(0,0)])
        self.assertEqual(g["distinct"],[])

    def test_loop_jump_and_missing_depth_abstain(self):
        a,b=view(1),view(2)
        b["c2w"][:3,3]+=5
        self.assertFalse(match_views(a,b)["matches"])
        a,b=view(1),view(2)
        b["points"]=[None,None]
        self.assertFalse(match_views(a,b)["matches"])

    def test_irregular_mapping_abstains_and_depth_projection_is_metric(self):
        s=snapshot(1)
        s["points"]=np.random.default_rng(1).normal(size=s["points"].shape)
        self.assertIsNone(metric_view(s,1.,8))
        xyz=surface_point(np.ones((30,40))*2,[16,10,23,20],[[100,0,20],[0,100,15],[0,0,1]],np.eye(4))
        self.assertAlmostEqual(xyz[2],2)
        self.assertLess(abs(xyz[0]),.1)


class NativeChecks(unittest.TestCase):
    def setUp(self):
        fixture=test_chain.ReportChainTests("test_canonical_memory_changes_after_observed_report_only")
        fixture.setUp()
        self.agent=a=fixture.make_agent(MultiviewFrontier)
        a.client.get_frame_points=Mock(return_value=[snapshot(1),snapshot(2)])
        a._map_step=None
        self.records={}
        for f in (1,2):
            for n,b in enumerate(view(f)["boxes"]):
                self.records[f,n]=a.instance_store.new_observation((0,.2,.5),"painting",frame_id=f,step=f,bbox=b)

    def test_veto_protects_neighbor_through_native_resolution(self):
        a=self.agent
        stored=a.instance_store.create_instance(self.records[1,0])
        with patch.dict(os.environ,{"NAV_COVISIBLE_GUARD":"1"}):
            self.assertFalse(a._distinct_at([self.records[2,0]],a._records(stored)))
            out=a._tool_resolve_duplicate(self.records[2,1].oid,"DUPLICATE",stored.iid)
            self.assertIn("error",out)
            self.assertIsNone(a.instance_store.instance_for_observation(self.records[2,1].oid))
            other=a.instance_store.create_instance(self.records[2,1])
            self.assertIn("error",a._tool_merge_instances(stored.iid,other.iid))
        self.assertEqual(a.instance_store.count_reported(),0)

    def test_proposal_coordinates_stay_on_live_frame(self):
        a=self.agent
        a._box_detector=Mock(return_value=Mock(detect=Mock(return_value=[dict(bbox=[25,22,55,59])])))
        with patch.dict(os.environ,{"NAV_LOCAL_BOXES":"1"}),patch.object(ReIDFrontier,"_build_decider_input",return_value=({},None)):
            state,_=a._build_decider_input(a._last_observation)
            state2,_=a._build_decider_input(a._last_observation)
        self.assertEqual(state["local_rgb_proposals"],state2["local_rgb_proposals"])
        self.assertNotIn("frame_id",state["local_rgb_proposals"])
        a._box_detector.return_value.detect.assert_called_once()
        self.assertEqual(len(a._proposal_queue),0)

    def test_real_pending_queue_only_and_independent_native_region(self):
        a=self.agent
        image=np.zeros((240,320,3),np.uint8)
        image[:120,:120]=255
        ok,jpeg=cv2.imencode(".jpg",image)
        self.assertTrue(ok)
        a.client.get_frame_image=lambda fid:({"found":True},jpeg.tobytes())
        snap=snapshot(1)
        snap["colors"]=image[::8,::8].reshape(-1,3)
        a.client.get_frame_points=lambda stride:[snap]
        a._spot_rejected=lambda *args:False
        a._proposal_limit=128
        a._record_crosshair_evidence=Mock()
        a.client.get_candidate_evidence=Mock(return_value=({"found":True},b"panel"))
        a._boxes=Mock(return_value=[dict(bbox=[80,80,144,160])])
        c=dict(refined=True,frame_id=1,pixel=[112,120],pixel_norm=[350,500],candidate_id="local1",bbox=[80,80,144,160])
        a.client.prepare_pixels=Mock(return_value=dict(candidates=[c]))
        with patch.dict(os.environ,{"NAV_LOCAL_BOXES":"1"}),patch.object(ReIDFrontier,"_tool_propose_candidates",return_value={"masks":[]}):
            out=a._tool_propose_candidates(1,"clock")
            self.assertEqual(out["local_pending_proposals"][0]["candidate_id"],"local1")
            self.assertEqual(a._proposal_queue["local1"]["status"],"pending")
            a._tool_propose_candidates(1,"clock")
        a.client.prepare_pixels.assert_called_once()
        self.assertEqual(a.instance_store.count_reported(),0)
        self.assertFalse(a.instance_store.nodes)

    def test_optional_pose_evidence_uses_native_duplicate_review_only(self):
        a=self.agent
        a.instance_store.create_instance(self.records[1,0])
        a._pose_hint=Mock(return_value={"score":.9,"threshold":.8})
        with patch.dict(os.environ,{"NAV_REID_POSE_WEIGHTS":"mock"}),patch.object(ReIDFrontier,"_dup_review_payload",return_value=([{"observation_id":3}],[],0)):
            rows,_,_=a._dup_review_payload()
        self.assertIn("pose_jepa_review",rows[0])
        self.assertEqual(len(a.instance_store.nodes),1)
        self.assertEqual(a.instance_store.count_reported(),0)


if __name__=="__main__":
    unittest.main()
