import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from delayed_pose import DelayedPoseMixin


class MiniAgent:
    def _init_report_memory(self):
        self._dup_ledger = {}
        self._instance_goal_index = {}
        self._pose_failed = False
        self.now = 1
        self.scale = None
        self.outcomes = []
        self.scored = 0
        self.node = NS(iid=4, reported=True)
        self.ref = NS(oid=1, frame_id=1, bbox=(0,0,12,12), step=0, evidence={})
        self.obs = NS(oid=2, frame_id=2, bbox=(0,0,12,12), step=1, point=(0,0,0), evidence={})
        self.owner = None
        self.instance_store = NS(get_observation=lambda x:self.obs if x==2 else None,
                                 get=lambda x:self.node, instance_for_observation=lambda x:self.owner,
                                 nearby=lambda *args,**kw:[(0.25,self.node)],
                                 observations={1:self.ref,2:self.obs})
        self.snapshots = {1:{"pose":[1.0]}, 2:{"pose":[1.0]}}
        self.instance_dup_radius_m = 0.6
        self.row = dict(observation_id=2, neighbors=[dict(instance_id=4,dist_m=0.1)], first_step=1)
        self._dup_ledger[2] = self.row

    def _audit(self, *args, **kw):
        pass

    def _metric_scale_value(self):
        return self.scale

    def _records(self, node):
        return [self.ref]

    def _map_snapshot(self):
        return self.snapshots

    def _pose_hint(self, row, weights):
        self.scored += 1
        return dict(instance_id=4, score=0.99)

    def _tool_commit_candidates(self, reviews, label=""):
        return dict(duplicate_review=[dict(self.row)], next="Inspect outstanding reviews.")

    def _build_decider_input(self, obs, local_map=False):
        return dict(duplicate_review_pending=list(self._dup_ledger.values())), []

    def _tool_resolve_duplicate(self, oid, decision, other=None, text=""):
        self.outcomes.append(decision)
        self._dup_ledger.pop(oid, None)
        return dict(resolved=decision)


class Agent(DelayedPoseMixin, MiniAgent):
    pass


class DelayTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, NAV_REID_POSE_WEIGHTS="fixture")
        env.start()
        self.addCleanup(env.stop)
        self.a = Agent()
        self.a._init_report_memory()

    def test_wait_then_scale(self):
        a=self.a
        out=a._tool_commit_candidates([])
        self.assertEqual(out["duplicate_review"], [])
        self.assertEqual(out["deferred_until_map_scale"], [2])
        self.assertEqual(a._tool_resolve_duplicate(2,"DUPLICATE",4)["status"],"waiting_for_map_scale")
        self.assertEqual(a._build_decider_input(None)[0]["duplicate_review_pending"],[])
        self.assertEqual(a.outcomes, [])
        a.now=2
        a.scale=2.5
        row,=a._build_decider_input(None)[0]["duplicate_review_pending"]
        self.assertEqual(row["pose_jepa_review"]["instance_id"],4)
        self.assertEqual(row["neighbors"][0]["dist_m"],0.25)
        self.assertEqual(a._dup_ledger[2]["neighbors"][0]["dist_m"],0.25)
        self.assertTrue(a.node.reported)
        self.assertEqual(a._tool_resolve_duplicate(2,"NEW")["resolved"],"NEW")
        self.assertEqual(a.scored,1)

    def test_no_scale_expiry(self):
        a=self.a
        a._tool_commit_candidates([])
        a.now=5
        row,=a._build_decider_input(None)[0]["duplicate_review_pending"]
        self.assertNotIn("dist_m",row["neighbors"][0])
        self.assertEqual(a._tool_resolve_duplicate(2,"DUPLICATE",4)["resolved"],"DUPLICATE")
        self.assertEqual(a.scored,0)

    def test_interleaved_scale_and_decision(self):
        a=self.a
        a._tool_commit_candidates([])
        a.scale=1.
        self.assertEqual(a._tool_resolve_duplicate(2,"DUPLICATE",4)["status"],"calibrated_review_available")
        self.assertEqual(a.outcomes,[])
        self.assertEqual(a._tool_resolve_duplicate(2,"NEW")["resolved"],"NEW")

    def test_already_resolved_not_reopened(self):
        a=self.a
        a._tool_commit_candidates([])
        a._dup_ledger.clear()
        a.owner=a.node
        a.scale=1.
        a._build_decider_input(None)
        self.assertEqual(a.scored,0)
        self.assertEqual(a._scale_waiting,{})
        self.assertTrue(a.node.reported)

    def test_same_view_distinct_and_session_reset(self):
        a=self.a
        a.obs.frame_id=1
        a.obs.bbox=(24,0,40,12)
        self.assertEqual(len(a._tool_commit_candidates([])["duplicate_review"]),1)
        self.assertEqual(a._tool_resolve_duplicate(2,"NEW")["resolved"],"NEW")
        a._init_report_memory()
        a._tool_commit_candidates([])
        self.assertEqual(set(a._scale_waiting),{2})
        a._init_report_memory()
        self.assertEqual(a._scale_waiting,{})

    def test_multiple_neighbors_no_wait(self):
        a=self.a
        a.row["neighbors"].append(dict(instance_id=5,dist_m=0.2))
        self.assertEqual(len(a._tool_commit_candidates([])["duplicate_review"]),1)
        self.assertFalse(a._scale_waiting)

    def test_new_remains_possible_before_scale(self):
        a=self.a
        a._tool_commit_candidates([])
        self.assertEqual(a._tool_resolve_duplicate(2,"NEW")["resolved"],"NEW")
        self.assertEqual(a._scale_waiting,{})

    def test_expired_neighbor_and_new_neighbor(self):
        a=self.a
        a._tool_commit_candidates([])
        a.scale=2.0
        second=NS(iid=5,reported=False)
        a.instance_store.nearby=lambda *args,**kw:[(0.22,second)]
        refused=a._tool_resolve_duplicate(2,"DUPLICATE",4)
        self.assertIn("error",refused)
        self.assertEqual(a._dup_ledger[2]["neighbors"][0]["instance_id"],5)
        self.assertEqual(a.outcomes,[])
        self.assertEqual(a._tool_resolve_duplicate(2,"DUPLICATE",5)["resolved"],"DUPLICATE")

    def test_neighbor_disappears_after_release(self):
        a=self.a
        a._tool_commit_candidates([])
        a.scale=1.0
        a._build_decider_input(None)
        a.instance_store.nearby=lambda *args,**kw:[]
        self.assertIn("error",a._tool_resolve_duplicate(2,"DUPLICATE",4))
        self.assertEqual(a._dup_ledger[2]["neighbors"],[])
        self.assertEqual(a._tool_resolve_duplicate(2,"NEW")["resolved"],"NEW")

    def test_prediction_cache_across_physical_steps(self):
        a=self.a
        a._tool_commit_candidates([])
        a.scale=1.0
        a._build_decider_input(None)
        for step in (3,4,5):
            a.now=step
            a._build_decider_input(None)
        self.assertEqual(a.scored,1)
        a.snapshots[2]["pose"]=[2.0]
        a.now=6
        a._build_decider_input(None)
        self.assertEqual(a.scored,2)


if __name__ == "__main__":
    unittest.main()
