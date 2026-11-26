"""Exercise the production report gate while a frozen-frame provider is pending."""
import json
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

from barrier_agent import BarrierAgent
from test_chain import ReportChainTests, REPORT, TURN


class StationaryTests(unittest.TestCase):
    def setUp(self):
        os.environ.update(NAV_STAGE3_IDENTITY="jepa", NAV_STAGE4_GEOMETRY="keyframes_only")
        fixture = ReportChainTests()
        fixture.setUp()
        self.fixture = fixture
        self.agent = fixture.make_agent(BarrierAgent)
        self.owner = fixture.register(self.agent, 3, (0., .5, .5))
        self.agent.target_instance_id = self.owner.iid
        self.agent._category = lambda _: "clock"
        self.agent._boxes = lambda *args: [dict(bbox=[80,80,240,240])]
        self.agent._recover_report = Mock(return_value=(TURN,"fresh_view"))
        self.agent._identity_observations = lambda *args: None

    def test_same_frame_report_blocks_then_waits_for_receipt(self):
        entered, release = threading.Event(), threading.Event()
        def delayed(fid, ref, obs, boxes):
            entered.set()
            if not release.wait(4):
                raise TimeoutError("test did not release provider")
            self.assertEqual(obs.step_count, 20)
            return dict(result="same",selected_box=0,reason="test_evidence")
        self.agent._stationary_verdict = delayed
        with ThreadPoolExecutor(max_workers=1) as executor:
            job = executor.submit(self.agent._report_found, self.owner.iid)
            try:
                self.assertTrue(entered.wait(4))
                self.assertFalse(job.done())
                self.agent._recover_report.assert_not_called()
                self.assertIsNone(self.agent._pending_report)
            finally:
                release.set()
            self.assertEqual(job.result(4), REPORT)
        self.assertFalse(self.owner.reported)
        self.agent._settle_report(self.fixture.observation(21,REPORT))
        self.assertTrue(self.owner.reported)
        self.assertEqual(self.agent._reported_count,1)
        self.agent._recover_report.assert_not_called()

    def test_worker_timeout_is_not_a_motion_or_model_failure(self):
        self.agent._stationary_verdict = Mock(side_effect=TimeoutError("test"))
        with self.assertRaisesRegex(RuntimeError,"before action release"):
            self.agent._report_found(self.owner.iid)
        self.agent._recover_report.assert_not_called()
        self.assertIsNone(self.agent._pending_report)

    def test_insufficient_evidence_keeps_clue_open(self):
        self.agent._stationary_verdict = Mock(return_value=dict(result="withheld",reason="no_depth"))
        self.assertEqual(self.agent._report_found(self.owner.iid),TURN)
        self.assertIsNone(self.agent._pending_report)
        self.assertFalse(self.owner.reported)
        self.assertIn(self.owner,self.agent.instance_store.available())

if __name__ == "__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(StationaryTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors))))
    raise SystemExit(not result.wasSuccessful())
