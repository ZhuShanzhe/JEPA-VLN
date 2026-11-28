"""Exercise the frozen real tracker and regional selector contract with scripted observations."""
from contextlib import ExitStack
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / d) for d in (
    "local_target_20261128", "approach_search_20261128", "gpt_inspect_only_20261128")]
import trigger_policy
from local_search import LocalTargetTracker
from commit_policy import CommittedInspectPolicy


class GrantTests(unittest.TestCase):
    def setUp(self):
        self.tracker = LocalTargetTracker()
        self.gate = Mock()
        self.policy = CommittedInspectPolicy(self.gate)
        self.memory = Mock()
        candidate = dict(prior=0.1, novelty=0.2, unseen=0.3, steps=4, blocked=False)
        self.memory.choose.return_value = (1, {"candidates": [candidate, candidate]})
        self.rows = [dict(position=(x,0), actions=[1], heading=0, key=k)
                     for x,k in ((1,"unused"),(5,"chosen"))]
        self.site = dict(patch_id=0, actions=[1,1], site_xy=(3.,0.), site_yaw=0., anchor_xy=(4.,0.),
                         surface="wall", support=.5)
        self.scores = [[.6]]
        self.route = dict(actions=[1]*4, status="active")
        observed = dict(score=[0.1,0.2], evidence=[0.3,0.4], current=[0,0,0],
                        patches=[dict(patch_id=0, bbox_norm=[0,0,1,1], surface="wall", support=.5)])
        ctx = ExitStack()
        self.addCleanup(ctx.close)
        hooks = dict(request=lambda _,mode,**kw: observed if mode=="observe" else self.scores,
                     project_patches=lambda *a,**kw:[self.site],
                     propose_sites=lambda *a,**kw:[self.site.copy()],
                     geometric_plan=lambda *a,**kw:self.route, visible_map_surfaces=lambda *a:set())
        for name, fn in hooks.items():
            ctx.enter_context(patch.object(trigger_policy, name, fn))

    def step(self, mode, budget, x=0., blocked=False):
        self.gate.choose.return_value = dict(mode=mode, region_id="r0", patch_id=0 if mode=="inspect" else None)
        d, r = self.policy.pick(self.tracker, self.memory, None, (x,0.,0.), 1., self.rows,
                                [None]*3, "clock", budget, None, blocked=blocked)
        self.assertIs(r, self.rows[1])
        self.assertEqual(d["old_selected_region_key"], d["region_key"])
        return d

    def test_explore_first_is_unmodified_region(self):
        result = self.step("explore", 100)
        self.assertEqual(result["action_source"], "region")
        self.assertIsNone(self.tracker.target)

    def test_advisory_explore_preserves_same_site_and_observed_arrival(self):
        start = self.step("inspect", 100)
        later = self.step("explore", 92, x=1.)
        self.assertEqual((start["goal_id"], start["site_xy"]), (later["goal_id"], later["site_xy"]))
        self.assertTrue(self.gate.choose.call_args.kwargs["memory"]["active_local"])
        self.assertEqual(later["exec_status"], "committed_view_retained")
        self.assertIsNone(later["released"])
        self.assertEqual(later["commit_actions_left"], 16)
        self.route = dict(actions=[], status="arrived")
        reached = self.step("explore", 85, x=3.)
        self.assertEqual(reached["released"]["event"], "arrived")
        self.assertIsNone(self.tracker.target)

    def test_unknown_geometry_stalled_and_stronger_visual_evidence(self):
        first = self.step("inspect", 100)
        self.site.update(anchor_xy=(0.,4.), site_xy=(0.,3.), support=.52)
        self.scores = [[.61]]
        weak = self.step("inspect", 94, x=.25)
        self.assertEqual(weak["goal_id"], first["goal_id"])
        self.site["support"] = .75
        self.scores = [[.85]]
        strong = self.step("inspect", 88, x=.5)
        self.assertNotEqual(strong["goal_id"], first["goal_id"])
        self.assertEqual(strong["released"]["event"], "stronger_new_support")
        self.assertEqual(strong["commit_actions_left"], 24)
        self.route = dict(actions=[], status="blocked", reason="obstacle")
        stopped = self.step("explore", 82, x=.75)
        self.assertEqual(stopped["released"]["event"], "no_reachable_route")

    def test_timeout_and_fast_path_count_intervening_scans(self):
        self.step("inspect", 100)
        left = self.step("explore", 77, x=1.)
        self.assertEqual(left["actions"], [1])
        self.assertEqual(left["commit_actions_left"], 1)
        self.assertIsNone(self.policy.fast_continue(self.tracker, None,(1.,0.,0.),1.,76))
        expired = self.step("explore", 72, x=1.25)
        self.assertEqual(expired["released"]["event"], "inspection_budget")
        self.assertEqual(expired["action_source"], "region")
        self.assertEqual(self.tracker.history[-1][1], "inspection_budget")
        self.assertIsNone(self.tracker.target)

    def test_repeated_contact_can_cancel(self):
        self.step("inspect", 100)
        self.step("explore", 94, blocked=True)
        stopped = self.step("explore", 88, blocked=True)
        self.assertEqual(stopped["released"]["event"], "blocked_without_progress")


if __name__ == "__main__":
    unittest.main()
