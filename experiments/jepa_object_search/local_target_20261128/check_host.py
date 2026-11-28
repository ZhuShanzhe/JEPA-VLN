"""Network-denied host integration with real HM3D RGB and a controlled observed map."""
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


def run(output):
    project = Path(os.environ['NAV_ROOT'])
    sys.path.insert(0,str(project/'jepa_exec_v14'))
    from verify_rank import deny_ip_sockets, make_scene
    deny_ip_sockets()
    from benchmark_api import Action
    from evaluation.main import run_eval
    from evaluation.main.evaluator import _reset_environment
    from check_policy import GridFixture
    from most_frontier_probe import ProbeAgent
    from local_agent import LocalApproachAgent

    env, ep, data = make_scene(project)
    try:
        _reset_environment(env,ep,data)
        sim_agent = env.sim.get_agent(0)
        names = {v.name:k for k,v in sim_agent.agent_config.action_space.items()}
        def rgb():
            frames = env.sim.get_sensor_observations()
            return next(v[..., :3].copy() for k,v in frames.items() if 'rgb' in k.lower())
        with tempfile.TemporaryDirectory() as td:
            os.environ.update(NAV_DEBUG_ROOT=td,NAV_RUN_ID='local_fixture',MAPPING_DEBUG_DIR=td)
            args = SimpleNamespace(seed=0,legacy_agent_api=False,log_actions=False,log_action_limit=0,
                                   save_api_images_dir=None)
            agent = run_eval.make_agent(os.environ['MOST_FROZEN_AGENT'],None,args)
            assert isinstance(agent,LocalApproachAgent)
            pose = [0.,0.,0.]
            scale = [1.]
            agent._frontier_scale = 1.
            agent._frontier_pose = tuple(pose)
            agent._frontier_server_revision = None
            agent._frontier_grid = GridFixture()
            # Extend the finite, controlled map to cover paths after area selection.
            g = agent._frontier_grid
            g.obstacle = np.zeros((241,241),bool)
            g.geometry_observed = np.ones_like(g.obstacle)
            g.origin = np.array([-12.,-12.])
            g.world_to_cell = lambda p:tuple(np.floor((np.asarray(p)-g.origin)/g.res).astype(int))
            g.in_bounds = lambda c:0<=c[0]<241 and 0<=c[1]<241
            agent._metric_scale_value = lambda:scale[0]
            agent._estimated_current_pose = lambda:tuple(pose)
            agent.client = SimpleNamespace(get_all_poses=lambda:(None,None))
            obs = SimpleNamespace(episode_id='local_fixture',step_count=0,goal_text='find all clocks',
                                  max_steps=300,previous_action=None,rgb=rgb())
            agent.output_dir = td
            def dispatch(a):
                a = int(a)
                assert a in (1,2,3)
                env.sim.step(names[{1:'move_forward',2:'turn_left',3:'turn_right'}[a]])
                if a==1:
                    pose[0]+=.25*math.cos(pose[2]); pose[1]+=.25*math.sin(pose[2])
                else:
                    pose[2]+=math.pi/6 if a==2 else -math.pi/6
                obs.step_count+=1
                agent.now=obs.step_count
                obs.previous_action=a; obs.rgb=rgb()
                agent._feed_frame(obs)
            def scan():
                endpoint = (pose[0]+3*math.cos(pose[2]),pose[1]+3*math.sin(pose[2]))
                cluster = dict(path=[pose[:2],endpoint],world=endpoint,scale=1.,key=endpoint,branch_id=None,
                               path_cost_m=3.,utility=1.)
                agent._last_frontier_clusters=[cluster]
                assert agent._activate_frontier(obs,cluster,'autonomous EXPLORE')
                if agent.scan is not None:
                    for expected in (3,2,2,3):
                        a=agent._explore_follow(obs)
                        assert a==expected,(a,expected)
                        dispatch(a)
                first=agent._explore_follow(obs)
                assert first in (1,2,3), first
                dispatch(first)
            def finish_leg():
                for _ in range(42):
                    a=agent._explore_follow(obs)
                    if a is None:
                        return
                    dispatch(a)
                raise AssertionError('observation carrier exceeded expected horizon')
            def events():
                return [json.loads(s) for s in (Path(td)/'v13_routes.jsonl').read_text().splitlines()]
            with patch.object(ProbeAgent,'_feed_frame',lambda *a:None):
                agent.init_episode(obs)
                agent._feed_frame(obs)
                try:
                    scan()
                    first=next(e for e in events() if e['event']=='local_route_choice')
                    assert first['action_source']=='region' and first['bound'],first
                    finish_leg()
                    assert agent.local.inside(pose), (agent.local.region,pose)
                    # Add observed walls to the controlled map upon arrival. No wall,
                    # class label, landmark or depth is supplied by the actual simulator.
                    yy,xx=np.indices(g.obstacle.shape)
                    dx=g.origin[0]+(xx+.5)*g.res-pose[0]
                    dy=g.origin[1]+(yy+.5)*g.res-pose[1]
                    g.obstacle[:] = np.abs(np.hypot(dx,dy)-3.1)<.11
                    scan()
                    second=[e for e in events() if e['event']=='local_route_choice'][-1]
                    assert second['transition']=='acquired' and second['bound'],second
                    finish_leg()
                    from map_views import visible_map_surfaces
                    # Isolate the unchanged-geometry condition, without injecting
                    # a visual score, candidate, motion plan or selected target.
                    agent.local.scan_surfaces=visible_map_surfaces(g,tuple(pose),1.)
                    scan()
                    third=[e for e in events() if e['event']=='local_route_choice'][-1]
                    assert third['transition']=='retained' and third['goal_id']==second['goal_id'],third
                    assert third['region_phase']=='retained_goal_geometry_check',third
                    assert second['site_xy']==third['site_xy'] and second['site_yaw']==third['site_yaw']
                    assert third['distance_m'] < second['distance_m'], (second,third)
                    # A real, native high-priority command must clear the local carrier.
                    finish_action=agent._execute_decider_result(obs,SimpleNamespace(action='FINISH'),'fixture')
                    assert finish_action==int(Action.FINISH) and agent._focus_view is None and agent.local.target is None
                    # A scale change must clear every coordinate-based memory.
                    agent.local.region=(1.,1.); scale[0]=1.12
                    assert agent._current_frame() is not None and agent.local.region is None
                    assert any(e['event']=='local_map_reset' for e in events())
                    receipt=dict(scene='HM3D',agent_factory=True,real_rgb=True,
                                 controlled_map_and_pose=True,network_denied=True,paid_requests=0,
                                 geometry_source=second['anchor_source'],site_id=second['goal_id'],
                                 site_xy=second['site_xy'],surface=second['surface'],
                                 acquired_distance_m=second['distance_m'],retained_distance_m=third['distance_m'],
                                 acquired_actions=second['actions'],retained_actions=third['actions'],
                                 phase=first['region_phase'],no_rescan_when_geometry_unchanged=True,
                                 native_finish_preempted=True,scale_reset=True)
                finally:
                    if agent.channel is not None:
                        agent.channel.close()
                        agent.worker.wait(timeout=45)
            Path(output).write_text(json.dumps(receipt,indent=2)+'\n')
            print(json.dumps(receipt))
    finally:
        env.close()


if __name__=='__main__':
    run(sys.argv[1])
