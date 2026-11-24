"""Synthetic long corridors with the actual MOST PathFollower. No model/API calls."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from agents.navigator import PathFollower
from delayed_forks import DeferredForks, trunk_plan


class NativeLoop:
    def _explore_follow(self, o):
        action, _ = self._explore_follower.next_action()
        if action is not None:
            self._explore_follower.dead_reckon(action)
        return action

    def _feed_frame(self, o):
        self.observed_step = self.now

    def _execute_decider_result(self, *args):
        return "native"

    def act(self, o):
        return 9  # competing report or stop


class Corridor(DeferredForks, NativeLoop):
    def __init__(self, count):
        ends = [(17, 0), (10, 8), (10, -8), (16, 6)][:count]
        self.clusters = [dict(key=(i, 1), world=p, path=[(0, 0), (10, 0), p], scale=1.)
                         for i, p in enumerate(ends)]
        self._last_frontier_clusters = list(self.clusters)
        self._frontier_pose = (0., 0., 0.)
        self._frontier_scale = 1.
        fl = PathFollower(scale=1., reach_m=.8)
        fl.set_path(self.clusters[0]["path"])
        fl.anchor_frame = 10
        self._explore_follower = fl
        self._frontier_revision = 1
        self._last_explore_plan = -10
        self._metric_snapshot = {"revision": 1}
        self._active_frontier_key = (0, 1)
        self.scan = self.deferred = None
        self._adjusting = self._scanning = self._last_motion_failed = False
        self.mode = "explore"
        self.pitch = self.yaw = self.now = self.observed_step = 0
        self.word = "clock.n.01"
        self.scan_actions = (3, 2, 2, 3)
        self.history = []
        self.events = []
        self.external_revision = (10, 1, 0)
        self._frontier_server_revision = (10, 1, 0)
        self.client = NS(get_state=lambda: self.external_revision)
        self._frontier_grid = NS(astar=self.astar, shortcut=lambda path: path)
        self.channel = NS(send=lambda msg: None, poll=lambda t: True, recv=lambda: {"index": 1})
        self.last_selection = None

    def astar(self, start, goal):
        return [tuple(start), (10., 0.), tuple(goal[:2])]

    def _plan_exploration(self, o, select=True):
        assert not select
        self._last_explore_plan = self.now
        self._frontier_revision += 1
        self._last_frontier_step = self.now
        # Original on cooldown. Only current accepted alternatives are selectable.
        self._last_frontier_clusters = self.clusters[1:]
        return True

    def _estimated_current_pose(self):
        fl = self._explore_follower
        return fl.x, fl.y, fl.yaw

    def _map_revision_tuple(self, state):
        return state

    def _begin_scan(self, original, ids, plans, source):
        self.scan = dict(ids=ids, plans=plans, yaw=self.yaw, views=[1, 2, 3], frame_steps=[],
                         carrier=self._explore_follower)

    def _end_scan(self, reason):
        self.scan = None

    def accept_selection(self, *args):
        self.last_selection = args[1:3]

    def emit(self, event, **fields):
        self.events.append((event, fields))

    def schedule(self):
        self._schedule_delay(NS(), 0, "deterministic explorer", 120)

    def confirm_step(self, blocked=False):
        action = self._explore_follow(NS(max_steps=180))
        self.now += 1
        self._last_motion_failed = blocked
        self._feed_frame(NS(previous_action=action))

    def reach_scan(self):
        self.schedule()
        for _ in range(len(self.deferred["prefix"])):
            self.confirm_step()
        self._start_deferred_scan(NS(max_steps=180), self.deferred)
        assert self.scan is not None


class CheckDeferred(unittest.TestCase):
    def test_two_and_four_distant_native_branches(self):
        for n in (2, 4):
            host = Corridor(n)
            host.reach_scan()
            self.assertGreater(host._estimated_current_pose()[0], 3.)
            self.assertEqual(len(host.scan["ids"]), n)
            self.assertEqual(host._last_explore_plan, -10)
            self.assertIsNone(host.deferred)
            def activate(agent, o, cluster, source):
                fl = PathFollower(scale=cluster["scale"], reach_m=.8)
                fl.set_path(cluster["path"])
                agent._explore_follower = fl
                return True
            with patch("most_frontier_probe.ProbeAgent._activate_frontier", activate):
                host._rank_after_scan(NS(max_steps=180), host.scan)
            self.assertEqual(host.last_selection, ("active", "f0"))
            self.assertEqual(sum(event == "rank" for event, _ in host.events), 1)

    def test_impossible_budget_and_repeated_geometry(self):
        h = Corridor(2)
        args = [h.clusters, 0, h._frontier_pose, h._explore_follower]
        self.assertIsNone(trunk_plan(*args, 24))
        h.clusters[1]["path"] = h.clusters[0]["path"]
        self.assertIsNone(trunk_plan(*args, 140))

    def test_interruptions(self):
        for event in ("collision", "priority", "replan", "decision", "pose"):
            h = Corridor(2)
            h.schedule()
            if event == "collision":
                h.confirm_step(blocked=True)
            elif event == "priority":
                h.act(NS())
            elif event == "replan":
                h._frontier_revision += 1
                h._explore_follow(NS())
            elif event == "decision":
                h._execute_decider_result(NS(), NS(action="REPORT"), None)
            else:
                for _ in range(len(h.deferred["prefix"])):
                    h.confirm_step()
                h._explore_follower.y += 1.
                h._start_deferred_scan(NS(max_steps=180), h.deferred)
            self.assertIsNone(h.deferred, event)
            self.assertIsNone(h.scan, event)

    def test_map_and_frontier_invalidation(self):
        for event in ("map_revision", "preexisting_loop_closure", "path_removed", "mapper_exception",
                      "no_alternative", "lost_follower"):
            h = Corridor(2)
            h.reach_scan()
            if event == "map_revision":
                h.external_revision = (11, 2, 1)
            elif event == "preexisting_loop_closure":
                h._frontier_server_revision = (11, 2, 1)
            elif event == "path_removed":
                h._frontier_grid.astar = lambda start, goal: None
            elif event == "mapper_exception":
                def failed(*_):
                    raise RuntimeError("mapper disconnected")
                h._frontier_grid.astar = failed
            elif event == "no_alternative":
                h.clusters = h.clusters[:1]
            else:
                h._explore_follower = None
            h._rank_after_scan(NS(max_steps=180), h.scan)
            self.assertIsNone(h.last_selection, event)
            self.assertFalse(any(e == "rank" for e, _ in h.events), event)


if __name__ == "__main__":
    unittest.main()
