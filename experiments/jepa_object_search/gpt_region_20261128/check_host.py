"""One real high-level call in the native exploration hook, with a controlled navigation grid."""
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import tempfile


def run(output):
    project = Path(os.environ["NAV_ROOT"])
    sys.path.insert(0,str(project/"jepa_exec_v14"))
    from verify_rank import make_scene
    from evaluation.main import run_eval
    from evaluation.main.evaluator import _reset_environment
    from most_frontier_probe import ProbeAgent
    from check_policy import GridFixture
    from gpt_agent import GPTRegionAgent
    from gpt_gate import MODEL
    env,ep,data=make_scene(project)
    try:
        _reset_environment(env,ep,data)
        names={v.name:k for k,v in env.sim.get_agent(0).agent_config.action_space.items()}
        def rgb():
            return next(im[:,:,:3].copy() for name,im in env.sim.get_sensor_observations().items() if "rgb" in name.lower())
        private=output.parent/"private_host"
        private.mkdir(mode=0o700,exist_ok=True)
        td=tempfile.mkdtemp(dir=private)
        with patch.dict(os.environ,dict(NAV_DEBUG_ROOT=td,NAV_RUN_ID="gpt_fixture",MAPPING_DEBUG_DIR=td)):
            args=SimpleNamespace(seed=0,legacy_agent_api=False,log_actions=False,log_action_limit=0,save_api_images_dir=None)
            agent=run_eval.make_agent(os.environ["MOST_FROZEN_AGENT"],None,args)
            assert isinstance(agent,GPTRegionAgent)
            agent.output_dir=td
            agent._frontier_scale=1.
            agent._frontier_pose=(0.,0.,0.)
            agent._frontier_server_revision=None
            agent._frontier_grid=GridFixture()
            agent._metric_scale_value=lambda:1.
            agent._estimated_current_pose=lambda:(0.,0.,agent.yaw*math.pi/6)
            agent.client=SimpleNamespace(get_all_poses=lambda:(None,None))
            endpoint=(3.,0.)
            cluster=dict(path=[(0.,0.),endpoint],world=endpoint,scale=1.,key=endpoint,branch_id=None,path_cost_m=3.,utility=1.)
            agent._last_frontier_clusters=[cluster]
            obs=SimpleNamespace(episode_id="gpt_exploration_fixture",step_count=0,goal_text="find all clocks",
                                max_steps=300,previous_action=None,rgb=rgb())
            with patch.object(ProbeAgent,"_feed_frame",lambda *args:None):
                agent.init_episode(obs)
                try:
                    agent._feed_frame(obs)
                    assert agent._activate_frontier(obs,cluster,"autonomous EXPLORE")
                    for expected in (3,2,2,3):
                        act=agent._explore_follow(obs)
                        assert act==expected
                        env.sim.step(names[{2:"turn_left",3:"turn_right"}[int(act)]])
                        obs.rgb=rgb()
                        obs.step_count+=1
                        agent.now=obs.step_count
                        obs.previous_action=act
                        agent._feed_frame(obs)
                    first_action=int(agent._explore_follow(obs))
                    assert first_action in (1,2,3)
                finally:
                    if agent.channel is not None:
                        agent.channel.close()
                        agent.worker.wait(timeout=45)
            events=[json.loads(l) for l in (Path(td)/"v13_routes.jsonl").read_text().splitlines()]
            choice,=[v for v in events if v["event"]=="local_route_choice"]
            assert choice["bound"] and choice["gpt_mode"] in ("explore","inspect")
            result=dict(factory=True,real_hm3d_rgb=True,controlled_grid_and_pose=True,
                        physical_scan_actions=4,mode=choice["gpt_mode"],action_source=choice["action_source"],
                        navigation_action_dispatched=True,provider_model=MODEL)
            output.write_text(json.dumps(result,indent=2)+"\n")
    finally:
        env.close()


if __name__=="__main__":
    run(Path(sys.argv[1]).resolve())
