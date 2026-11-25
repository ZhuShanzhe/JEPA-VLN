"""Offline checks using native canonical instances and native report accounting."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import cv2
import numpy as np

from verify_rank import deny_ip_sockets
deny_ip_sockets()
from agents.instance_store import InstanceMemory
from benchmark_api import Action
from confirm_agent import ConfirmFrontier
from view_memory import evidence_patch,live_match
from view_check import *  # include the causal frontier tests in the signed preflight


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(303)
        self.rgb=rng.integers(10,245,size=(320,320,3),dtype=np.uint8)
        ok,enc=cv2.imencode(".jpg",cv2.cvtColor(self.rgb,cv2.COLOR_RGB2BGR))
        assert ok
        a=self.a=ConfirmFrontier.__new__(ConfirmFrontier)
        a._views={}
        a._retry_after={}
        a._endorsed=set()
        a._attempts={}
        a._confirm_pending=None
        a._confirm_frame=None
        a._confirm_orient=0
        a._unreachable_instance_ids=set()
        a.now=20
        a._arrival_transition_active=False
        a._last_observation=SimpleNamespace(rgb=self.rgb,step_count=20)
        a._reported_count=0
        a._instance_goal_index={}
        a._goal_found=set()
        a._scanning=False
        a.report_near_dist_m=1.
        a._note=Mock()
        a._log_event=Mock()
        a._stamp_pending_outcome=Mock()
        a._clear_current_target=Mock()
        a._instance_distance_m=lambda node:.6
        a._geometry=lambda node,fid:(.2,.7,.6,0.)
        a.client=SimpleNamespace(get_frame_pose=lambda f:np.eye(4),get_frame_image=lambda f:({"found":True},enc.tobytes()))
        store=a.instance_store=InstanceMemory()
        rec=store.new_observation((0,.5,.5),"goal",dict(pixel_norm=[500,500]),frame_id=3)
        self.node=store.create_instance(rec)
        a.target_instance_id=self.node.iid
        a._collect_views()
        self.assertEqual(len(a._views),1)
        self.assertIn(self.node.iid,a._endorsed)

    def test_matched_and_report_once(self):
        a=self.a
        self.assertEqual(a._report_found(self.node.iid),int(Action.TARGET_FOUND))
        self.assertTrue(self.node.reported)
        self.assertNotEqual(a._report_found(self.node.iid),int(Action.TARGET_FOUND))
        self.assertEqual(a._reported_count,1)

    def test_absent_or_far_cannot_claim(self):
        a=self.a
        a._return=lambda *args:None
        a._last_observation.rgb=np.zeros_like(self.rgb)
        self.assertNotEqual(a._report_found(self.node.iid),int(Action.TARGET_FOUND))
        a._last_observation.rgb=self.rgb
        a._geometry=lambda *args:(4.,3.,4.,0.)
        self.assertNotEqual(a._report_found(self.node.iid),int(Action.TARGET_FOUND))
        self.assertEqual(a._reported_count,0)

    def test_mapped_reference_drives_return(self):
        a=self.a
        a._schedule_revisit=Mock(return_value={"navigating":True})
        a._close_commit=a._end_scan=a._end_route=a._cancel_delay=lambda *_:None
        a._tool_view_instance=lambda _:None
        a._nav_action=lambda obs:(int(Action.MOVE_FORWARD),False,False)
        self.assertEqual(a._return(a._last_observation,[self.node]),int(Action.MOVE_FORWARD))
        a._schedule_revisit.assert_called_once_with(3)
        self.assertEqual(a._confirm_pending,self.node.iid)
        self.assertIsNone(a._return(a._last_observation,[self.node]))


if __name__=="__main__":
    import view_check
    suite=unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromTestCase(ConfirmationTests),
                              unittest.defaultTestLoader.loadTestsFromModule(view_check)))
    run=unittest.TextTestRunner(verbosity=2).run(suite)
    if not run.wasSuccessful():
        raise SystemExit(1)
    folder=Path(__file__).resolve().parent
    names=("confirm_agent.py","view_memory.py","view_explore.py","confirm_check.py","view_check.py")
    info=dict(passed=True,no_paid_calls=True,tests=run.testsRun,
              sha256={str(folder/n):hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in names})
    (folder/"no_api_check.json").write_text(json.dumps(info,indent=2)+"\n")
