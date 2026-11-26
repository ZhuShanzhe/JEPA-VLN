import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock,patch
import cv2
import numpy as np
from active_agent import ActiveFrontier
from candidate_scout import CandidateScoutMixin
import test_chain


class Policy(CandidateScoutMixin,ActiveFrontier):
    pass


class ScoutTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_chain.ReportChainTests()
        self.fixture.setUp()
        a=self.a=self.fixture.make_agent(Policy)
        img=np.full((72,96,3),85,dtype=np.uint8)
        self.obs=SimpleNamespace(rgb=img,step_count=20,goal_text="Find three color pictures depicting landscapes")
        self.jpeg=cv2.imencode(".jpg",img)[1].tobytes()
        a._scout_spec={"mode":"detector","threshold":.61}
        a._ev_deciding=True
        a._frame_ids_from_feed=lambda:(10,6)
        a._scout_images={2:np.full((16,16,3),85,np.uint8),5:np.full((16,16,3),85,np.uint8),
                         7:np.full((16,16,3),85,np.uint8)}
        a._last_feed_info={}
        a.client.get_all_poses=lambda:(None,[2,4,5,7])
        a.client.get_frame_image=lambda f:({"found":True},self.jpeg)
        a._pending_decision_images=[]
        a._note=Mock()
        a._boxes=Mock(return_value=[{"score":.8,"bbox":[5,10,60,55]}])
        a._scout_attribute_worker=Mock()
        a._scout_attribute_worker.margin.return_value=.03
        self.addCleanup(patch.stopall)
        patch.dict(os.environ,{"NAV_SOURCE_SCOUT":"1","NAV_LOCAL_BOXES":"1"}).start()
        patch.object(ActiveFrontier,"_build_decider_input",return_value=({"task":"native"},None)).start()

    def test_visible_mapped_sources_and_native_tool_contract(self):
        a=self.a
        state,_=a._build_decider_input(self.obs)
        detail=state["source_frame_scout"]
        self.assertEqual([r["frame_id"] for r in detail["views"]],[5,2])
        self.assertEqual(detail["category_proposal"],"painting")
        self.assertEqual(a._scout_attribute_worker.margin.call_args.args[1],self.obs.goal_text)
        self.assertFalse(a.instance_store.available())
        self.assertEqual([k for k,_ in a._pending_decision_images],["scout_frame_5"])
        a._build_decider_input(self.obs)
        self.assertEqual(len(a._pending_decision_images),1)
        prepared={"frame_id":5,"masks":[{"mask_id":1}]}
        with patch.object(ActiveFrontier,"_tool_propose_candidates",return_value=prepared) as native:
            self.assertEqual(a._tool_propose_candidates(5,self.obs.goal_text),prepared)
            native.assert_called_once_with(5,self.obs.goal_text)
        self.obs.step_count+=1
        self.assertEqual([r["frame_id"] for r in a._build_decider_input(self.obs)[0]["source_frame_scout"]["views"]],[2])

    def test_hard_negative_no_offer(self):
        self.a._scout_attribute_worker.margin.return_value=-.05
        self.assertNotIn("source_frame_scout",self.a._build_decider_input(self.obs)[0])
        self.assertFalse(self.a._pending_decision_images)

    def test_disabled_agent_never_reads_selector(self):
        with patch.dict(os.environ,{"NAV_SOURCE_SCOUT":"0"}), patch.object(Path,"read_text",side_effect=AssertionError("unexpected selector read")):
            self.a._init_report_memory()
            self.a._ev_deciding=True
            self.assertEqual(self.a._build_decider_input(self.obs)[0],{"task":"native"})
            self.assertIsNone(self.a._scout_spec)

    def test_load_configured_selector_and_missing_optional_file(self):
        with TemporaryDirectory() as temp:
            file=Path(temp)/"test.json"
            with patch.dict(os.environ,{"NAV_SCOUT_SELECTOR":str(file)}):
                self.a._scout_spec=None
                self.assertEqual(self.a._build_decider_input(self.obs)[0],{"task":"native"})
                self.assertTrue(self.a._scout_offline)
                file.write_text('{"mode":"detector","threshold":0.61}',encoding="utf-8")
                self.a._scout_offline=False
                self.assertIn("source_frame_scout",self.a._build_decider_input(self.obs)[0])

    def test_scene_rgb_mismatch_and_unmapped_sources(self):
        self.a._scout_images[2][:]=0
        self.a._scout_images[5][:]=255
        self.assertNotIn("source_frame_scout",self.a._build_decider_input(self.obs)[0])
        self.a._boxes.assert_not_called()

    def test_preview_and_missing_model_preserve_original_decision(self):
        self.a._ev_deciding=False
        self.assertEqual(self.a._build_decider_input(self.obs)[0],{"task":"native"})
        self.a._boxes.assert_not_called()
        self.a._ev_deciding=True
        self.a._scout_attribute_worker.margin.side_effect=RuntimeError("encoder unavailable")
        self.assertEqual(self.a._build_decider_input(self.obs)[0],{"task":"native"})
        self.assertTrue(self.a._scout_offline)


if __name__=="__main__":unittest.main()
