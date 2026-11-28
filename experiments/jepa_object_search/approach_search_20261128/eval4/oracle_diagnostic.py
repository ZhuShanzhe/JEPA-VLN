"""Post-evaluation geometry diagnostic; not imported by policies or training."""
import gzip
import json
import math
import numpy as np
import habitat_sim as hs
from habitat_sim.utils.common import quat_from_coeffs
from habitat_sim.errors import GreedyFollowerError
from physical_panel import sensor_sim, policy_pose, DepthGrid, frontier_plans, candidates, HERE, DATA


def target_views(ep):
    from stopwatch.eval.geometry import _load_goal_sets, load_scene_instances
    ids=_load_goal_sets(DATA)['scoring_class_ids']['hssdov'][ep['object_category']]
    return {r['object_id']:np.float32([vp['agent_state']['position'] for vp in r['view_points']])
            for r in load_scene_instances(DATA/'protocol/instance_viewpoints/hssd','val',ep['scene_id']) if r['class_id'] in ids and r['view_points']}


def near(sim,views):
    from stopwatch.eval.geometry import geodesic_distance
    pos=np.asarray(sim.get_agent(0).state.position)
    hit=set()
    for iid,pts in views.items():
        shortlist=pts[np.linalg.norm(pts-pos,axis=1)<.25001]
        if len(shortlist) and geodesic_distance(sim,pos,shortlist)<.25:hit.add(iid)
    return hit


def oracle(sim,views):
    follower=sim.make_greedy_follower(0,.16,stop_key='stop',forward_key='move_forward',left_key='turn_left',right_key='turn_right')
    goal=None; reached=set();unavailable=set();steps=0
    while steps<=300:
        unseen=near(sim,views)-reached
        if unseen:reached.update(unseen);goal=None
        if len(reached)>=2 or steps==300:break
        if goal is None:
            distances=[]
            for iid,points in views.items():
                if iid in reached:continue
                for j,pt in enumerate(points):
                    if (iid,j) in unavailable:continue
                    path=hs.ShortestPath()
                    path.requested_start=np.asarray(sim.get_agent(0).state.position);path.requested_end=pt
                    if sim.pathfinder.find_path(path) and math.isfinite(path.geodesic_distance):
                        distances.append((path.geodesic_distance,iid,j))
            if not distances:break
            _,iid,j=min(distances);goal=(iid,j);follower.reset()
        iid,j=goal
        try:act=follower.next_action_along(views[iid][j])
        except GreedyFollowerError:act='stop'
        if act=='stop':
            unavailable.add(goal);goal=None;continue
        sim.step(act);steps+=1
    return dict(reached=len(reached),quota_met=len(reached)>=2,steps=steps)


def candidate_gate(sim,views):
    st=sim.get_agent(0).get_state()
    pose=policy_pose(sim);grid=DepthGrid(pose[:2]);grid.update(pose,sim.get_sensor_observations()['depth'])
    for act in ['turn_right','turn_left','turn_left','turn_right']:
        obs=sim.step(act);pose=policy_pose(sim);grid.update(pose,obs['depth'])
    start=sim.get_agent(0).get_state()
    initial=near(sim,views)
    basic=frontier_plans(grid,pose,1.,296)
    for turn in (0,-1,1,-2,2):
        rotation=[2 if turn>0 else 3]*abs(turn)
        for n in (8,3,1):basic.append((f'depth/{turn}/{n}',rotation+[1]*n))
    rows=candidates(grid,pose,1.,basic,296)
    valid=0;max_new=0
    for row in rows:
        sim.get_agent(0).set_state(start)
        hit=set(initial)
        # Same nominal candidate endpoints, sensing on the real simulator.
        for act in row['actions']:
            obs=sim.step({1:'move_forward',2:'turn_left',3:'turn_right'}[act])
            hit.update(near(sim,views))
            if obs.get('collided',False) and act==1:break
        n=len(hit-initial)
        valid+=bool(n);max_new=max(max_new,n)
    sim.get_agent(0).set_state(st)
    return dict(initial_candidates=len(rows),candidates_with_new_instance=valid,max_new=max_new)


def main():
    output=[]
    for row in json.loads((HERE/'panel.json').read_text()):
        shard=DATA/'eval/hssdov/content'/(row['scene']+'.json.gz')
        ep=next(e for e in json.load(gzip.open(shard))['episodes'] if str(e['episode_id'])==str(row['id']))
        sim=sensor_sim(ep)
        try:
            st=hs.AgentState();st.position=np.float32(ep['start_position']);st.rotation=quat_from_coeffs(ep['start_rotation'])
            sim.get_agent(0).set_state(st);targets=target_views(ep)
            gate=candidate_gate(sim,targets)
            sim.get_agent(0).set_state(st)
            from stopwatch.eval.geometry import geodesic_distance
            initial=[geodesic_distance(sim,st.position,v) for v in targets.values()]
            output.append(dict(scene=row['scene'],goal=row['goal'],**gate,oracle=oracle(sim,targets),
                               start_in_one_m=sum(d<1 for d in initial),start_in_half_m=sum(d<.5 for d in initial),
                               start_nearest_geo_m=min(initial)))
            print(output[-1],flush=True)
        finally:sim.close()
    (HERE/'diagnostic.json').write_text(json.dumps(output,indent=2))

if __name__=='__main__':main()
