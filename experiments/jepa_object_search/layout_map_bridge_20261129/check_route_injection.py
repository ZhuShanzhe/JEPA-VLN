"""Paired integration test: identical real sensor packet, fresh vs stale upstream obstacle update."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT/"layout_detour_v2_20261129"),str(ROOT/"reactive_view_20261129")]
from grid_policy import GridPlanner
from layout_paths import path,axes
from reactive_route import RoutePlanner
from prior_cues_score import visual_input
from most_layout_bridge import MostLayoutBridge


def run():
    torch.set_num_threads(2)
    route=GridPlanner(RoutePlanner.load())
    episodes=0
    tested=changed=alternate=survey=stale_restores=0
    for sample in sorted((ROOT/"reactive_view_20261129/private_recovery_comparison").glob("*.npz")):
        with np.load(sample) as d:
            episode=str(d["house"])+"/"+str(d["source"])
            history=[dict(x=float(p[0]),z=float(p[1]),yaw=float(p[2]),rgb=rgb,depth=depth,bump=bool(b),
                          episode=episode,floor=0,frame_id=i)
                     for i,(p,rgb,depth,b) in enumerate(zip(d["pose"],d["rgb"],d["depth"],d["bump"]))]
            box=d["box"].copy()
            remaining=int(d["remaining"])
        start,now=history[0],history[-1]
        episodes+=1
        x,y,r,b=map(int,box)
        ctx=route.previous.encoder.encode([visual_input(start["rgb"]),visual_input(start["rgb"][y:b,x:r])]).ravel()
        def act():
            return tuple(route.choose(ctx,box,start["depth"],history,remaining)["actions"])
        original=act()
        if 1 not in original:
            continue
        tested+=1
        xy=np.array([now["x"],now["z"]])
        def world_points(plan):
            return path(plan)@axes(now["yaw"])+xy
        pos=world_points(original)[0]
        # Test obstacle is a controlled counterfactual in the exact native snapshot schema.
        # The original archived room and its geometry labels are never used as a policy input.
        size=257
        res=.125
        origin=np.repeat(-16.0625,2)
        u,v=np.meshgrid(np.arange(size)*res+origin[0]+res/2,
                        np.arange(size)*res+origin[1]+res/2)
        obstacle=(u-pos[1])**2+(v+pos[0])**2 < .16**2
        grid=SimpleNamespace(free=np.zeros_like(obstacle),geometry_observed=obstacle,obstacle=obstacle,
                             origin=origin,res=res,unit_per_m=1.)
        agent=SimpleNamespace(episode_id=episode,pose_free=True,_last_frontier_step=len(history)-1,
                              _frontier_grid=grid,_frontier_pose=(xy[1],-xy[0],now["yaw"]),
                              _frontier_scale=1.,_frontier_revision=6)
        step=now["frame_id"]
        observation=SimpleNamespace(episode_id=episode,step_count=step)
        patch=MostLayoutBridge().snapshot(agent,observation,(*xy,now["yaw"]),refresh=False)
        assert patch.query(*pos)==2
        now["layout_patch"]=patch
        replacement=act()
        changed+=int(replacement!=original)
        if 1 in replacement:
            pts=world_points(replacement)
            assert np.all(patch.query(pts[:,0],pts[:,1])!=2)
            alternate+=1
        else:
            survey+=1
        now["frame_id"]=step+1
        stale_restores+=int(act()==original)
    assert tested>0 and tested==changed==stale_restores
    report=dict(frozen_real_sensor_packets=episodes,base_with_translation=tested,
                changed_after_fresh_upstream_obstacle=changed,
                new_translation_avoiding_blocked_cells=alternate,turn_or_stop=survey,
                stale_map_restores_original=stale_restores,
                scope="Interface stress test on stored RGB-D and measured odometry. A synthetic obstacle at the first translation tests causal map use. Not a new performance score.")
    (HERE/"route_injection_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)


if __name__=="__main__":
    with threadpool_limits(limits=2):
        run()
