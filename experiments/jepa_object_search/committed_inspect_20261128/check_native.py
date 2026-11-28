"""Native MOST factory and physical HM3D smoke test with scripted GPT intentions."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
from commit_agent import CommittedInspectAgent
import gpt_agent


class ScriptedGate:
    choices = []

    def __init__(self, *_):
        pass

    def choose(self, views, goal, regions, patches, memory):
        mode = "explore" if not self.choices else "inspect"
        chosen = max(patches, key=lambda p:p["support"])
        self.choices.append(mode)
        return dict(mode=mode, patch_id=chosen["patch_id"] if mode=="inspect" else None,
                    region_id=regions[-1]["id"], cue="controlled fixture", target_evidence="weak")


def main():
    fixture = HERE.parent / "local_target_20261128/check_host.py"
    spec = importlib.util.spec_from_file_location("local_hm3d_host_fixture", fixture)
    host = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(host)
    factory_class = CommittedInspectAgent
    original = factory_class._execute_decider_result
    budget = {}

    def exercise_at_finish(agent, obs, result, event, images=None):
        if result.action == "FINISH":
            # The standard fixture reaches this point immediately after issuing a
            # primitive on the retained carrier. Exercise its real dispatch routine
            # at the synthetic elapsed-budget boundary without moving the simulator.
            active = agent._focus_view
            assert active and active[1].startswith("local/") and agent.local.target is not None
            saved = (obs.step_count, agent._explore_follower)
            start_remaining = agent._gated.start_remaining
            obs.step_count = obs.max_steps-start_remaining+agent._gated.horizon
            action = agent._explore_follow(obs)
            assert action is None and agent._explore_follower is None and agent._focus_view is None
            budget.update(synthetic_clock_at_expiry=True, expired_carrier_emitted_no_action=True)
            # Restore the actual, still-live fixture moment to exercise native
            # priority on an active site independently of the clock-boundary check.
            obs.step_count, agent._explore_follower = saved
            agent._focus_view = active
        return original(agent, obs, result, event, images)

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "fixture.json"
        with patch.dict(os.environ, {"NAV_REQUIRE_DECISION_PREFLIGHT":"0"}), patch.object(gpt_agent, "GPTGate", ScriptedGate), patch.object(
                factory_class, "_execute_decider_result", exercise_at_finish):
            host.run(dest)
        receipt = json.loads(dest.read_text())
    assert ScriptedGate.choices == ["explore","inspect"]
    out = {k:receipt[k] for k in ("scene","agent_factory","real_rgb","controlled_map_and_pose",
           "network_denied","paid_requests","acquired_distance_m","retained_distance_m",
           "no_rescan_when_geometry_unchanged","native_finish_preempted","scale_reset")}
    out.update(scripted_upper_intents=ScriptedGate.choices, trained_local_scorer=True, **budget)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
