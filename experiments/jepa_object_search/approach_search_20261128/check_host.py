"""Factory and live decision-to-action test with a real HM3D camera and a controlled map."""
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

project = Path(os.environ["NAV_ROOT"])
sys.path.insert(0, str(project / "jepa_exec_v14"))
from verify_rank import deny_ip_sockets, make_scene
from check_policy import GridFixture


def run(output):
    deny_ip_sockets()
    from evaluation.main import run_eval
    from evaluation.main.evaluator import _reset_environment
    from most_frontier_probe import ProbeAgent
    from search_agent import PriorApproachAgent
    env, ep, data = make_scene(project)
    try:
        _reset_environment(env, ep, data)
        sim_agent = env.sim.get_agent(0)
        names = {v.name: k for k,v in sim_agent.agent_config.action_space.items()}
        def rgb():
            frames = env.sim.get_sensor_observations()
            return next(v[..., :3].copy() for k,v in frames.items() if "rgb" in k.lower())
        with tempfile.TemporaryDirectory() as td:
            os.environ.update(NAV_DEBUG_ROOT=td, NAV_RUN_ID="prior_fixture", MAPPING_DEBUG_DIR=td)
            args = SimpleNamespace(seed=0, legacy_agent_api=False, log_actions=False, log_action_limit=0,
                                   save_api_images_dir=None)
            agent = run_eval.make_agent("search_agent:PriorApproachAgent", None, args)
            assert isinstance(agent, PriorApproachAgent)
            agent._frontier_scale = 1.
            agent._frontier_pose = (0., 0., 0.)
            agent._frontier_server_revision = None
            agent._frontier_grid = GridFixture()
            agent._metric_scale_value = lambda: 1.
            agent._estimated_current_pose = lambda: (0., 0., agent.yaw * math.pi / 6)
            agent.client = SimpleNamespace(get_all_poses=lambda: (None, None))
            endpoint = (3., 0.)
            cluster = dict(path=[(0., 0.), endpoint], world=endpoint, scale=1., key=(3, 0), branch_id=None,
                           path_cost_m=3., utility=1.)
            agent._last_frontier_clusters = [cluster]
            obs = SimpleNamespace(episode_id="prior_fixture", step_count=0, goal_text="find all clocks",
                                  max_steps=300, previous_action=None, rgb=rgb())
            agent.output_dir = td
            moves = []
            with patch.object(ProbeAgent, "_feed_frame", lambda *a: None):
                agent.init_episode(obs)
                agent._feed_frame(obs)
                assert agent._activate_frontier(obs, cluster, "autonomous EXPLORE")
                try:
                    for n in range(5):
                        a = agent._explore_follow(obs)
                        assert a in (1, 2, 3)
                        moves.append(int(a))
                        env.sim.step(names[{1:"move_forward",2:"turn_left",3:"turn_right"}[int(a)]])
                        obs.step_count = agent.now = n + 1
                        obs.previous_action, obs.rgb = a, rgb()
                        agent._feed_frame(obs)
                finally:
                    if agent.channel is not None:
                        agent.channel.close()
                        agent.worker.wait(timeout=30)
            events = [json.loads(s) for s in (Path(td) / "v13_routes.jsonl").read_text().splitlines()]
            choice, = [e for e in events if e["event"] == "prior_route_choice"]
            assert moves[:4] == [3,2,2,3] and choice["bound"] and choice["phase"] == "prior_search"
            assert len(choice["candidates"]) > 1 and any(e["event"] == "focus_view_start" for e in events)
            receipt = dict(scene="HM3D", agent_factory=True, real_rgb=True, local_pretrained_prior=True,
                           controlled_map_and_pose=True, paid_requests=0, scan_actions=moves[:4],
                           first_dispatched_action=moves[4], selected=choice["selected"],
                           candidate_count=len(choice["candidates"]), phase=choice["phase"])
            Path(output).write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(receipt))
    finally:
        env.close()


if __name__ == "__main__":
    run(sys.argv[1])
