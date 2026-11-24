"""Offline contract checks for JEPA's handoff to an exploration follower.

These tests exercise the production adapter. The host returns deterministic
follower actions and acknowledges simulated actuator commands. No model, scene
assets, network connection, or paid decision client is used by this suite.
"""

import importlib.util
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class FixtureFollower:
    """Keep proposing a different action even after reaching the nominal fork."""

    def __init__(self):
        self.proposals = []
        self.anchor_frame = -1
        self.x = self.y = self.yaw = 0.
        self.scale = 1.
        self.arrived = False

    def next_action(self):
        self.proposals.append(1)
        return (None, True) if self.arrived else (1, False)


class FixtureHost:
    def __init__(self):
        self.now = 0
        self.episode = None
        self.scan = self.route = None
        self.events = []
        self._explore_follower = FixtureFollower()
        self._last_motion_failed = False
        self.observed_step = -1

    def emit(self, event, **fields):
        self.events.append(dict(event=event, step=self.now, **fields))

    def init_episode(self, observation):
        self.episode = str(observation.episode_id)
        self.now = observation.step_count

    def _explore_follow(self, observation):
        command, arrived = self._explore_follower.next_action()
        fl = self._explore_follower
        if command == 1:
            fl.x += .25 * math.cos(fl.yaw)
            fl.y += .25 * math.sin(fl.yaw)
        elif command in (2, 3):
            fl.yaw += math.radians(30) * (1 if command == 2 else -1)
        return None if arrived else command

    def _execute_decider_result(self, observation, result, event, images=None):
        assert result.action == "REPORT_FOUND"
        return 6

    def _feed_frame(self, observation):
        self._last_motion_failed = observation.blocked
        self.observed_step = observation.step_count

    def act(self, observation):
        self.now = observation.step_count
        self._feed_frame(observation)
        return self._explore_follow(observation)


def load_adapter():
    """Load a fresh adapter against the small host for this test process only."""
    parent = ModuleType("jepa_pilot_agent")
    parent.JepaFrontier = FixtureHost
    api = ModuleType("benchmark_api")
    api.Action = SimpleNamespace(FINISH=0, MOVE_FORWARD=1, TURN_LEFT=2, TURN_RIGHT=3,
                                 LOOK_UP=4, LOOK_DOWN=5, TARGET_FOUND=6)
    adapter_path = Path(__file__).with_name("jepa_exec_agent.py")
    spec = importlib.util.spec_from_file_location("jepa_exec_agent", adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"jepa_pilot_agent": parent, "benchmark_api": api,
                                      spec.name: adapter}):
        spec.loader.exec_module(adapter)
    return adapter.JepaExecutionFrontier


class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent_type = load_adapter()

    def setUp(self):
        self.agent = self.agent_type()
        self.obs = SimpleNamespace(episode_id="clock-fixture", step_count=0,
                                   max_steps=300, previous_action=None, blocked=False)
        self.agent.init_episode(self.obs)

    def select(self, native, selected, *, same_id=False):
        self.agent.accept_selection(self.obs, "f0", "f0" if same_id else "f1",
                                    list(native), list(selected))

    def dispatch(self):
        return self.agent._explore_follow(self.obs)

    def ack(self, command, *, blocked=False):
        self.obs.previous_action = command
        self.obs.blocked = blocked
        self.obs.step_count += 1
        self.agent.now = self.obs.step_count
        self.agent._feed_frame(self.obs)

    def ends(self):
        return [row["reason"] for row in self.agent.events if row["event"] == "commit_end"]

    def test_immediate_fork_requires_receipt(self):
        self.select([1, 1], [2, 1])
        self.assertEqual(self.dispatch(), 2)
        self.assertNotIn("first_difference_observed", self.ends())
        self.ack(2)
        self.assertEqual(self.ends(), ["first_difference_observed"])
        self.assertEqual(self.dispatch(), 1)

    def test_long_common_prefix_clock_regression(self):
        prefix = [1, 1, 2, 1, 1, 3, 1, 1, 1, 1]
        self.select(prefix + [1, 1], prefix + [3, 1])
        for command in prefix:
            self.assertEqual(self.dispatch(), command)
            self.ack(command)
            self.assertNotIn("first_difference_observed", self.ends())
        self.assertEqual(self.dispatch(), 3)
        self.ack(3)
        self.assertEqual(self.ends(), ["first_difference_observed"])

    def test_blocked_first_different_action(self):
        self.select([2, 1], [1, 1])
        self.assertEqual(self.dispatch(), 1)
        self.ack(1, blocked=True)
        self.assertEqual(self.ends(), ["forward_blocked"])

    def test_contradictory_receipt_aborts_without_success(self):
        self.select([1, 1], [2, 1])
        self.assertEqual(self.dispatch(), 2)
        self.ack(3)
        self.assertEqual(self.ends(), ["command_not_acknowledged"])
        self.assertEqual(self.dispatch(), 1)

    def test_report_preempts_long_common_prefix(self):
        self.select([1, 1, 1], [1, 1, 2])
        self.assertEqual(self.dispatch(), 1)
        self.ack(1)
        report = self.agent._execute_decider_result(
            self.obs, SimpleNamespace(action="REPORT_FOUND", target_id="fixture"), "fixture"
        )
        self.assertEqual(report, 6)
        self.assertEqual(self.ends(), ["native_decision_REPORT_FOUND"])
        self.assertEqual(self.dispatch(), 1)

    def test_identical_route_never_claims_execution(self):
        self.select([2, 1], [2, 1], same_id=True)
        self.assertEqual(self.dispatch(), 1)
        self.ack(1)
        self.assertNotIn("first_difference_observed", self.ends())

    def test_new_episode_releases_carrier(self):
        self.select([1, 1], [3, 1])
        self.obs.episode_id = "unseen-fixture"
        self.agent.init_episode(self.obs)
        self.assertEqual(self.dispatch(), 1)
        self.assertEqual(self.ends(), ["episode_reset"])

    def test_short_budget_does_not_bind_unreachable_fork(self):
        self.obs.max_steps = 2
        self.select([1, 1, 1], [1, 1, 2])
        self.assertIsNone(self.agent.commit)
        bypass, = [row for row in self.agent.events if row["event"] == "commit_bypass"]
        self.assertEqual(bypass["reason"], "insufficient_steps")

    def test_final_dispatched_command_without_receipt_is_pending(self):
        self.obs.max_steps = 1
        self.select([1], [3])
        self.assertEqual(self.agent.act(self.obs), 3)
        self.assertIn("commit_waiting_final_observation", [e["event"] for e in self.agent.events])
        self.assertEqual(self.ends(), [])
        self.ack(3)
        self.assertEqual(self.ends(), ["first_difference_observed"])

    def test_material_pose_correction_preempts(self):
        self.select([1, 1], [1, 2])
        self.assertEqual(self.dispatch(), 1)
        self.ack(1)
        self.agent._explore_follower.anchor_frame = 2
        self.agent._explore_follower.x += 1.
        self.assertEqual(self.dispatch(), 1)
        self.assertEqual(self.ends(), ["map_pose_corrected"])

    def test_arrival_before_branch_releases_control(self):
        self.select([1, 1], [1, 2])
        self.agent._explore_follower.arrived = True
        self.assertIsNone(self.dispatch())
        self.assertEqual(self.ends(), ["native_path_ended"])

    def test_replacing_native_follower_restores_original_callback(self):
        old = self.agent._explore_follower
        self.select([1, 1], [2, 1])
        self.agent._explore_follower = FixtureFollower()
        self.assertEqual(self.dispatch(), 1)
        self.assertEqual(old.next_action(), (1, False))
        self.assertEqual(self.ends(), ["native_follower_replaced"])

    def test_secondary_consumer_of_same_observation_does_not_abort(self):
        self.select([1, 1], [1, 2])
        self.assertEqual(self.dispatch(), 1)
        self.ack(1)
        self.agent._feed_frame(self.obs)
        self.assertEqual(self.dispatch(), 2)
        self.ack(2)
        self.assertEqual(self.ends(), ["first_difference_observed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
