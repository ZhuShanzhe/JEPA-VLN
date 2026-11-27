"""Context, rival identity and action-receipt tests for the additional report path."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from approach_agent import ApproachAgent
from benchmark_api import Action
from test_chain import ReportChainTests as Fixture
from transfer_agent import ArrivalTransferAgent


class GateTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Fixture()
        self.fixture.setUp()
        a = self.a = self.fixture.make_agent(ArrivalTransferAgent)
        self.n = self.fixture.register(a,3,(0.,.5,.5))
        self.ref = SimpleNamespace(roi=(.25,.25,.75,.75))
        a._references = lambda n: [(3,self.ref)]
        a._collect_views = lambda: None
        a._boxes = Mock(return_value=[dict(bbox=[80,80,240,240])])
        a._category = lambda o: "painting"
        a._competing_identity = Mock(return_value=False)
        a._capture_report_view = lambda: None
        a._recover_report = Mock(return_value=(int(Action.TURN_LEFT),"relook"))
        a._arrival_matches = Mock(return_value={0:(3,self.ref)})
        a.target_instance_id = self.n.iid
        patcher=patch.object(ApproachAgent,"_report_gate",return_value=(False,"identity_missing_geometry"))
        self.prior = patcher.start()
        self.addCleanup(patcher.stop)

    def invoke(self,event="arrival"):
        decision = SimpleNamespace(action="REPORT_FOUND",target_id=str(self.n.iid))
        return self.a._execute_decider_result(self.a._last_observation,decision,event)

    def test_native_decision_and_receipt(self):
        a=self.a
        self.assertEqual(self.invoke(),int(Action.TARGET_FOUND))
        self.assertIsNone(a._arrival_request)
        self.assertFalse(self.n.reported)
        a._settle_report(self.fixture.observation(21,int(Action.TARGET_FOUND)))
        self.assertTrue(self.n.reported)
        self.assertEqual(a._confirmed_reference[self.n.iid],(3,self.ref))

    def test_no_historical_vote_for_automatic_probe(self):
        self.assertFalse(self.a._report_gate(self.n,self.a._last_observation)[0])
        self.a._arrival_matches.assert_not_called()

    def test_side_report_requires_proximity(self):
        self.a.target_instance_id=99
        self.a._instance_distance_m=lambda n:2.
        self.assertNotEqual(self.invoke(),int(Action.TARGET_FOUND))
        self.a._arrival_matches.assert_not_called()

    def test_same_step_and_instance_only(self):
        self.a._arrival_request=(19,self.n.iid,"arrival",True)
        self.assertFalse(self.a._report_gate(self.n,self.a._last_observation)[0])
        self.a._arrival_request=(20,self.n.iid+1,"arrival",True)
        self.assertFalse(self.a._report_gate(self.n,self.a._last_observation)[0])

    def test_do_not_override_different_or_uncertain(self):
        for state in ["identity_different","identity_uncertain","other_instance_explains_same_live_region"]:
            self.prior.return_value=False,state
            self.assertNotEqual(self.invoke(),int(Action.TARGET_FOUND))
        self.a._arrival_matches.assert_not_called()

    def test_ambiguous_owner_or_empty_detection(self):
        self.a._boxes.return_value=[]
        self.assertNotEqual(self.invoke(),int(Action.TARGET_FOUND))
        self.a._boxes.return_value=[dict(bbox=[80,80,240,240])]
        other=self.fixture.register(self.a,9,(1.,.2,.3))
        with patch("transfer_agent.relation",return_value="unknown"):
            self.assertNotEqual(self.invoke(),int(Action.TARGET_FOUND))
        self.assertFalse(other.reported or self.n.reported or self.a._pending_report)


if __name__=="__main__":
    unittest.main()
