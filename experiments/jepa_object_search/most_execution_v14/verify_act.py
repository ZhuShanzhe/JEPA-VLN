"""Exercise the public MOST act dispatcher on HM3D with injected high-level events."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from verify_rank import deny_ip_sockets, make_scene


def run_case(env, episode, data, case, output):
    import numpy as np
    from agents.navigator import PathFollower
    from benchmark_api import Action
    from evaluation.main.evaluator import _reset_environment
    from jepa_exec_agent import JepaExecutionFrontier
    from most_frontier_probe import ProbeAgent

    _reset_environment(env, episode, data)
    native = env.sim.get_agent(0)
    names = {s.name: k for k, s in native.agent_config.action_space.items()}
    ids = {1: "move_forward", 2: "turn_left", 3: "turn_right"}

    def rgb():
        frames = env.sim.get_sensor_observations()
        return np.asarray(next(v for k, v in frames.items() if "rgb" in k.lower()))[..., :3].copy()

    agent = JepaExecutionFrontier.__new__(JepaExecutionFrontier)
    noop = lambda *args, **kwargs: None
    for name in ("_process_queued_picks", "_settle_action_outcomes", "_capture_goal_images",
                 "_ensure_caption_task", "_capture_pool_world_anchor", "_periodic_anchor",
                 "_log_event", "_record_and_update", "_write_probe"):
        setattr(agent, name, noop)
    agent.vlm = None
    agent.channel = agent.worker = agent.follower = None
    agent._adjusting = agent._last_motion_failed = agent._scanning = False
    agent._finish_check_action = agent.target_instance_id = None
    agent._active_branch_key = None
    agent._last_feed_info = agent._metric_snapshot = {}
    agent._paper_steps = []
    agent._reported_count = 0
    agent.mode = "explore"
    agent.explore_enabled = agent.pose_free = True
    agent._frontier_pose = (0., 0., 0.)
    agent._explore_recovery_queue = []
    agent._explore_recovery_stage = 0
    agent._active_frontier_key = "fixture"
    agent._frontier_failures = {}
    agent._last_caption_check_step = -1
    agent._caption_hit_max_seen = -1
    agent.caption_hit_interval = 1
    agent.scale_source = "fixture"
    agent.decision_loop = object()
    agent.output_dir = str(output)
    fl = PathFollower(1., reach_m=.35)
    fl.path = [(0.,0.), (8.,0.)]
    agent._explore_follower = fl
    # Actual caption polling / control transfer remain active. Only the
    # external result, report store and VLM decision are replaced.
    agent.client = SimpleNamespace(
        get_all_poses=lambda: (None, None),
        get_captioned_frame_ids=lambda: (case == "report" and agent.now == 1, [5]))
    agent._caption_hit_in_new_frames = lambda fids: (5, None, "fixture")
    agent._report_found = lambda *args: int(Action.TARGET_FOUND)
    agent._choose_high_level_target = lambda o, event: agent._execute_decider_result(
        o, SimpleNamespace(action="REPORT_FOUND", target_id="fixture"), event)
    agent._should_finish = lambda o: case == "finish" and o.step_count == 1
    obs = SimpleNamespace(step_count=0, max_steps=300, episode_id="public-act-offline",
                          rgb=rgb(), previous_action=None, target_mode="all",
                          target_count=3, goal_text="find all lamps")
    watched = {"act", "_feed_frame", "_explore_action", "_caption_hit_decision", "_explore_follow",
               "_dispatch", "_execute_decider_result", "_close_commit"}
    modules = {"jepa_exec_agent", "jepa_pilot_agent", "most_frontier_probe", "nav_agent"}
    calls = []
    def record(frame, event, arg):
        module = Path(frame.f_code.co_filename).stem if event == "call" and frame.f_code.co_name in watched else None
        if module in modules:
            pair = module + "." + frame.f_code.co_name
            if pair not in calls:
                calls.append(pair)

    dispatched = []
    motion = []
    with patch.object(ProbeAgent, "_feed_frame", noop):
        agent.init_episode(obs)
        assert agent.accept_selection(obs, "f0", "f1", [1,2], [1,3])
        try:
            sys.setprofile(record)
            for tick in range(3 if case == "follow" else 2):
                action = int(agent.act(obs))
                dispatched.append(action)
                if action not in ids or tick == 2:
                    break
                before = native.get_state()
                env.sim.step(names[ids[action]])
                after = native.get_state()
                moved = float(np.linalg.norm(after.position-before.position))
                rotation = float(abs(np.dot(after.rotation.components, before.rotation.components)))
                blocked = action == 1 and bool(env.sim.previous_step_collided)
                assert not blocked and (moved > .001 or rotation < .9999)
                motion.append(dict(forward_m=moved, cos_half_turn=rotation))
                obs.previous_action = action
                obs.step_count += 1
                obs.rgb = rgb()
                agent._last_motion_failed = blocked
        finally:
            sys.setprofile(None)
    assert dispatched[0] == int(Action.MOVE_FORWARD)
    assert agent.commit is None
    rows = [json.loads(line) for p in output.glob("*.jsonl") for line in p.read_bytes().splitlines()]
    dispatch = [r["action"] for r in rows if r["event"] == "commit_dispatch"]
    end, = [r for r in rows if r["event"] == "commit_end"]
    assert {"nav_agent.act", "nav_agent._explore_action", "nav_agent._explore_follow",
            "jepa_exec_agent.act", "jepa_exec_agent._dispatch"} <= set(calls)
    if case == "follow":
        assert dispatch == [1,3] and dispatched[:2] == [1,3]
        assert end["reason"] == "first_difference_observed" and end["confirmed_steps"] == 2
    elif case == "report":
        assert dispatched == [1,int(Action.TARGET_FOUND)] and dispatch == [1]
        assert end["reason"] == "native_decision_REPORT_FOUND" and end["confirmed_steps"] == 1
        assert {"nav_agent._caption_hit_decision", "nav_agent._execute_decider_result"} <= set(calls)
    else:
        assert dispatched == [1,int(Action.FINISH)] and dispatch == [1]
        assert end["reason"] == "higher_priority_action" and end["confirmed_steps"] == 1
    return dict(case=case, returned_by_public_act=dispatched, commit_dispatch=dispatch,
                end=end["reason"], confirmed=end["confirmed_steps"], observed_motions=motion,
                called_methods=calls)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("select a new output path")
    deny_ip_sockets()
    sys.path.insert(0, str(args.project / "jepa_env_v13/most_integration"))
    env, episode, data = make_scene(args.project)
    cases = []
    try:
        for case in ("follow", "report", "finish"):
            with tempfile.TemporaryDirectory() as temp:
                cases.append(run_case(env, episode, data, case, Path(temp)))
    finally:
        env.close()
    result = dict(passed=True, network="kernel IP socket denial", paid_requests=0,
                  scene="HM3D branch_execution_03", cases=cases)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(cases))


if __name__ == "__main__":
    main()
