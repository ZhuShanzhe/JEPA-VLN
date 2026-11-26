"""Original MOST tool/registry interfaces, fixed sensor and mapper replies; no API."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from candidate_evidence import CandidateEvidenceMixin
from decision.vlm import VLMDecisionClient
from repair_agent import RepairFrontier
from summarize_candidate_evidence import summarize
import test_chain


class LiveEvidence(CandidateEvidenceMixin, RepairFrontier):
    pass


class CandidateChecks(unittest.TestCase):
    def setUp(self):
        f = test_chain.ReportChainTests()
        f.setUp()
        self.f = f
        self.a = a = f.make_agent(LiveEvidence)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        a.output_dir = self.tmp.name
        a._queued_picks, a._queued_pick_results, a._pending_decision_images = {}, [], []
        a._proposal_limit = 64
        a._frame_ids_from_feed = lambda:(300, 300)
        a._record_crosshair_evidence = Mock()
        a._record_rejected_spot = Mock()
        a._dup_review_payload = lambda:([], [], 0)
        a.instance_dup_radius_m = 3
        a._aligned_point = lambda arr:arr
        a._schedule_revisit = Mock()
        a.client.get_candidate_evidence = lambda c:({"found":True}, f.jpeg)
        self.not_ready = {"error":"candidate registration is not available yet (frame 1 is already captured)",
                          "error_code":"FRAME_NOT_READY"}

    def menu(self, masks=(1,2)):
        out = dict(frame_id=1,masks=[dict(mask_id=i,area_frac=.1) for i in masks],_tool_images=[("som_1",self.f.jpeg)])
        with patch.object(RepairFrontier, "_tool_propose_candidates", return_value=out):
            self.a._tool_propose_candidates(1,"painting")

    def mapped(self, masks=(1,2)):
        return {"candidates":[dict(candidate_id="c"+str(i),frame_id=1,mask_id=i,bbox=[10,25,55,80],
                                   pixel_norm=[200+50*i,500]) for i in masks]}

    def events(self):
        return [json.loads(s) for s in (Path(self.tmp.name)/"candidate_evidence.jsonl").read_text().splitlines()]

    def test_native_retry_reoffer_and_positive_registration(self):
        a = self.a
        self.menu()
        a.client.pick_segment = Mock(side_effect=[self.not_ready,self.not_ready,self.mapped()])
        self.assertTrue(a._tool_pick_segment(1,[1,2],"painting")["queued"])
        a._process_queued_picks()
        self.assertTrue(a._tool_pick_segment(1,[1,2],"painting")["queued"])
        a._process_queued_picks()
        self.assertEqual(a.client.pick_segment.call_count, 2)
        a._last_observation.step_count += 4
        a._process_queued_picks()
        self.assertEqual([x.args for x in a.client.pick_segment.call_args_list],[(1,[1,2])]*3)
        first = a._queued_pick_results[0]["out"]
        self.assertEqual(len(first["proposals"]),2)
        state = {"queued_pick_results":[dict(frame_id=1,proposals=first["proposals"])]}
        a._ev_offer_pending(state,24)
        a._ev_offer_pending({},25)
        duplicate = {}
        a._ev_offer_pending(duplicate,25)
        self.assertEqual(duplicate,{})
        self.assertEqual(dict(a._pending_decision_images)["proposal_c1"],self.f.jpeg)
        self.assertEqual(len(a.instance_store.available()),0)
        a.client.resolve_candidates = Mock(return_value={"candidates":{"c1":{"found":True,"point":[0,0.1,0.6]}}})
        out = a._tool_commit_candidates([dict(candidate_id="c1",verdict="ACCEPT"),
                                         dict(candidate_id="c2",verdict="REJECT")],"painting")
        self.assertEqual([r["candidate_id"] for r in out["instances"]],["c1"])
        self.assertEqual(out["rejected"],["c2"])
        a._ev_offer_pending({},27)
        replay = a._tool_pick_segment(1,[1,2],"painting")
        self.assertEqual(a.client.pick_segment.call_count,3)
        self.assertEqual([r["status"] for r in replay["proposals"]],["active","reject"])
        self.assertEqual(len(a.instance_store.available()),1)

    def test_changed_mask_table_and_permanent_error_are_not_retried(self):
        a = self.a; self.menu()
        a.client.pick_segment = Mock(side_effect=[self.not_ready,self.not_ready])
        a._tool_pick_segment(1,[1],"painting")
        a._process_queued_picks()
        self.menu((6,))
        self.assertFalse(a._ev_delayed)
        self.assertTrue(a._queued_pick_results[-1]["failed"])
        self.assertIn("region_version_changed",[r["event"] for r in self.events()])
        a.client.pick_segment = Mock(side_effect=[self.not_ready,{"error":"mask not found","error_code":"MASK_NOT_FOUND"}])
        a._tool_pick_segment(1,[6],"painting")
        a._process_queued_picks()
        self.assertFalse(a._ev_delayed)

    def test_missing_depth_and_uncertain_do_not_register(self):
        a = self.a; self.menu()
        a.client.pick_segment = Mock(return_value=self.mapped())
        a._tool_pick_segment(1,[1,2],"painting")
        a.client.resolve_candidates = Mock(return_value={"candidates":{"c1":{"found":False,"error":"no valid depth"}}})
        out = a._tool_commit_candidates([dict(candidate_id="c1",verdict="ACCEPT"),
                                         dict(candidate_id="c2",verdict="UNCERTAIN")],"painting")
        self.assertEqual(len(a.instance_store.available()),0)
        self.assertEqual(out["geometry_rejections"][0]["candidate_id"],"c1")
        self.assertEqual(out["uncertain"],["c2"])

    def test_actual_transport_receipts_and_reset(self):
        a = self.a; self.menu()
        a.client.pick_segment = Mock(return_value=self.mapped())
        a._tool_pick_segment(1,[1,2],"painting")
        path = Path(a.output_dir)/"vlm_calls.jsonl"
        source = dict(kind="decision",ok=True,call_id="s1",context={"step":21},images=[{"label":"proposal_c2"}])
        path.write_text(json.dumps(source)+"\n"+json.dumps(dict(source,ok=False,images=[{"label":"proposal_c1"}]))+"\n")
        a._ev_client_cursor = 0
        a._ev_read_client_receipts(path)
        self.assertEqual(a._ev_confirmed_panels,{"c2"})
        a._ev_read_client_receipts(path)
        a._ev_offer_pending({},22)
        self.assertEqual(a._pending_decision_images[-1][0],"proposal_c1")
        core = [(label,self.f.jpeg) for label in ("current_observation","topdown_map","selected_candidate")]
        with patch.dict(os.environ,{"NAV_VLM_MAX_IMAGES":"4"}):
            sent = VLMDecisionClient._image_parts(core + a._pending_decision_images)
        self.assertEqual([entry[0] for entry in sent], [row[0] for row in core]+["proposal_c1"])
        self.assertEqual(sum(r["event"]=="successful_client_panel" for r in self.events()),1)
        a._init_report_memory()
        self.assertFalse(a._ev_confirmed_panels or a._ev_delayed or a._ev_selections or a._ev_candidates)

    def test_previews_do_not_consume_single_delivery(self):
        a = self.a
        receipt = dict(frame_id=1, out=dict(proposals=[dict(candidate_id="c1",mask_id=1)]))
        a._queued_pick_results.append(receipt)
        def native_snapshot(_self, observation, local_map=False):
            q = [dict(frame_id=r["frame_id"],proposals=r["out"]["proposals"]) for r in a._queued_pick_results]
            a._queued_pick_results = []
            return dict(queued_pick_results=q),None
        with patch.object(RepairFrontier, "_build_decider_input",native_snapshot):
            state,_ = a._build_decider_input(a._last_observation)
            self.assertFalse(state["queued_pick_results"])
            self.assertEqual(a._queued_pick_results,[receipt])
            a._ev_deciding = True
            state,_ = a._build_decider_input(a._last_observation)
            self.assertEqual(state["queued_pick_results"][0]["proposals"][0]["candidate_id"],"c1")
            self.assertFalse(a._queued_pick_results)

    def test_missing_evidence_only_retries_original_candidate(self):
        a = self.a; self.menu()
        a.client.pick_segment = Mock(return_value=self.mapped((1,)))
        a.client.get_candidate_evidence = Mock(side_effect=[({"found":False},b""),({"found":True},self.f.jpeg)])
        a._tool_pick_segment(1,[1],"painting")
        state = {}
        a._ev_offer_pending(state,22)
        self.assertEqual(a.client.get_candidate_evidence.call_args_list[0].args,("c1",))
        self.assertEqual(a.client.get_candidate_evidence.call_args_list[1].args,("c1",))
        self.assertEqual(state["queued_pick_results"][0]["proposals"],[{"candidate_id":"c1","mask_id":1}])
        self.assertFalse(a.instance_store.available())

    def test_v23_genuine_paint_history(self):
        pilot = Path(os.environ["NAV_ROOT"])/"mapping_runtime/pilots/jepa_reid_v23_first2"
        episode = json.loads((pilot/"panel.json").read_text())["rows"][1]["episode_id"]
        receipt = json.loads((pilot/"comparison/accepted/second"/(episode+".json")).read_text())
        got = summarize(Path(receipt["result"]).parent/"agent")
        expected = dict(region_propose_calls=9,pick_calls=7,ever_queued_frames=7,native_async_completed_batches=6,
                        native_async_proposal_rows=16,native_mapping_not_ready_delivered_frames=1,
                        queued_state_distinct_candidates=13,successful_client_distinct_panels=9,commit_calls=0)
        for key,n in expected.items():
            self.assertEqual(got[key],n,key)
