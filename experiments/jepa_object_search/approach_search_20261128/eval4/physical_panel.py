"""Online RGB-D only navigation. StopWatch instance viewpoints score the finished trajectory separately."""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import habitat_sim as hs
from habitat_sim.utils.common import quat_from_coeffs
import magnum as mn
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT / "a_hssdov_external"
sys.path.insert(0,str(HERE/'locked'))
from search_policy import PriorSearchMemory
from enhanced_search import PriorEvidenceMemory
from map_views import candidates, visible_map_surfaces
from global_frontiers import frontier_plans
from point_executor import ViewFollower


class DepthGrid:
    """Per-episode map from projected, horizon-level depth rays; no navmesh access."""
    res = .125

    def __init__(self,origin):
        self.origin = np.array(origin) - 40.
        self.geometry_observed = np.zeros((640,640),bool)
        self.obstacle = self.geometry_observed.copy()
        self.blocked = self.geometry_observed.copy()

    def world_to_cell(self,p):
        return tuple(np.floor((np.asarray(p)-self.origin)/self.res).astype(int))

    def in_bounds(self,c):
        return 0<=c[0]<640 and 0<=c[1]<640

    def traversable(self,c):
        if not self.in_bounds(c):
            return False
        x,y = c
        return bool(self.geometry_observed[y,x] and not self.blocked[y,x] and not self.obstacle[max(0,y-1):y+2,max(0,x-1):x+2].any())

    def record_contact(self,previous):
        # The simulator refused this forward step; persist its destination, not
        # the unvisited farther waypoint that the agent was originally seeking.
        x,y,a = previous
        for dist in (.20,.25,.30):
            cx,cy=self.world_to_cell((x+dist*math.cos(a),y+dist*math.sin(a)))
            if self.in_bounds((cx,cy)):
                self.blocked[cy,cx]=True

    def update(self,pose,depth):
        x,y,yaw = pose
        h,w = depth.shape
        # A band around chest height estimates clearance. Depth is optical-axis
        # distance; move in horizontal map coordinates using the pinhole rays.
        col=np.arange(0,w,3)
        horizontal = (col+.5-w/2)/(w/2) # 90 degree HFOV
        rows=np.arange(h//4,h,4)
        sampled=depth[rows[:,None],col[None,:]]
        height=1.31+(h/2.-rows[:,None]-.5)*sampled/(w/2.)
        hits=(sampled>.05)&(height>.12)&(height<1.41)
        depth=np.min(np.where(hits,sampled,20.),axis=0)
        # Free ray length is bounded by the farthest observed lower camera ray.
        # A ray directed into absent depth is not evidence of free floor.
        valid=np.where((sampled>.05)&(height<1.41),sampled,0.)
        measured=np.max(valid,axis=0)
        dx = math.cos(yaw)+horizontal*math.sin(yaw)
        dy = math.sin(yaw)-horizontal*math.cos(yaw)
        ds=np.arange(0.,5.01,.07)
        reach=(ds[:,None] <= np.minimum(np.minimum(5.,measured),depth-.10)[None,:])
        xs=((x+ds[:,None]*dx-self.origin[0])/self.res).astype(int)
        ys=((y+ds[:,None]*dy-self.origin[1])/self.res).astype(int)
        ok=reach & (xs>=0)&(xs<640)&(ys>=0)&(ys<640)
        self.geometry_observed[ys[ok],xs[ok]]=True
        ex=((x+depth*dx-self.origin[0])/self.res).astype(int)
        ey=((y+depth*dy-self.origin[1])/self.res).astype(int)
        ok=(depth<5.) & (ex>=0)&(ex<640)&(ey>=0)&(ey<640)
        self.geometry_observed[ey[ok],ex[ok]]=True
        self.obstacle[ey[ok],ex[ok]]=True
        cx,cy=self.world_to_cell((x,y))
        self.geometry_observed[cy-1:cy+2,cx-1:cx+2]=True
        self.obstacle[cy,cx]=False
        self.blocked[cy,cx]=False


def policy_pose(sim):
    state=sim.get_agent(0).state
    from quaternion import rotate_vectors
    fwd=rotate_vectors(state.rotation,[0,0,-1])
    return float(state.position[0]),float(-state.position[2]),math.atan2(-fwd[2],fwd[0])


def sensor_sim(ep):
    sys.path.insert(0,str(PROJECT/'src/stopwatch'))
    from stopwatch.data import semantics as sem
    from functools import lru_cache
    if not hasattr(sem,"_real_templates"):
        sem._real_templates=sem._hssd_templates((PROJECT/'simulation/hssd').resolve())
        sem._hssd_templates=lambda _:sem._real_templates
    basedir=DATA/'runtime_check/scenes'
    sem.prepare_scene(ep,basedir,DATA/'vocab')
    sc=basedir/ep['scene_id']
    c=hs.SimulatorConfiguration()
    c.scene_dataset_config_file=str(basedir/ep['scene_dataset_config'])
    c.scene_id=str(sc);c.enable_physics=True;c.allow_sliding=True
    a=hs.agent.AgentConfiguration(height=1.41,radius=.17)
    for name,kind in [('rgb',hs.SensorType.COLOR),('depth',hs.SensorType.DEPTH)]:
        s=hs.CameraSensorSpec()
        s.uuid=name;s.sensor_type=kind;s.hfov=90;s.resolution=[192,192];s.position=mn.Vector3(0,1.31,0)
        a.sensor_specifications.append(s)
    a.action_space={k:hs.agent.ActionSpec(k,hs.agent.ActuationSpec(amount=v)) for k,v in [('move_forward',.25),('turn_left',30.),('turn_right',30.)]}
    sim=hs.Simulator(hs.Configuration(c,[a]))
    file=PROJECT/'src/stopwatch/stopwatch/eval/env.py'
    fn=next(n for n in ast.parse(file.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_install_navmesh')
    scope={'EvalConfig':object}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(file),'exec'),scope)
    adapter=SimpleNamespace(pathfinder=sim.pathfinder,recompute_navmesh=sim.recompute_navmesh,
                            reconfigure=sim.reconfigure,habitat_config=SimpleNamespace(scene=str(sc)))
    scope['_install_navmesh'](adapter,SimpleNamespace(dataset=SimpleNamespace(navmesh_max_climb=.1,navmesh_include_static_objects=True)))
    return sim


def measure(sim,ep,positions):
    from stopwatch.eval.geometry import _load_goal_sets, load_scene_instances, geodesic_distance
    ids=_load_goal_sets(DATA)['scoring_class_ids']['hssdov'][ep['object_category']]
    instances=load_scene_instances(DATA/'protocol/instance_viewpoints/hssd','val',ep['scene_id'])
    targets={r['object_id']:np.float32([v['agent_state']['position'] for v in r['view_points']])
             for r in instances if r['class_id'] in ids and r['view_points']}
    initial_distances=[geodesic_distance(sim,positions[0],pts) for pts in targets.values()]
    nearest={k:float("inf") for k in targets}
    discovered={}
    seen=set()
    for step,pos in enumerate(positions):
        p=np.float32(pos)
        key=tuple(np.round(p,5))
        if key in seen:continue
        seen.add(key)
        for instance,views in targets.items():
            if instance in discovered:continue
            shortlist=views[np.linalg.norm(views-p,axis=1)<min(1.0001,nearest[instance]+.00001)]
            if len(shortlist):
                nearest[instance]=min(nearest[instance],geodesic_distance(sim,p,shortlist))
            if nearest[instance]<.25:
                discovered[instance]=step
    return dict(instances_available=len(targets),distinct_approached=len(discovered),
                already_at_start=sum(v==0 for v in discovered.values()),first_steps=sorted(discovered.values()),
                quota_met=len(discovered)>=2,reachable_instances_at_start=sum(math.isfinite(d) for d in initial_distances),
                approached_half_m=sum(d<.5 for d in nearest.values()),approached_one_m=sum(d<1. for d in nearest.values()))


class Worker:
    def __init__(self):
        py=PROJECT.parent/'jepa_instance_bench_20261016/policy_env/bin/python'
        self.proc=subprocess.Popen([str(py),'-u',str(HERE/'inference_worker.py'),str(PROJECT)],
                                   stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        if json.loads(self.proc.stdout.readline())!={"ready":True}:raise RuntimeError("model startup")
        self.tmp=tempfile.TemporaryDirectory(dir=HERE)
        self.views=Path(self.tmp.name)/'current.npy'

    def score(self,views,rows,goal,arm):
        np.save(self.views,np.stack(views),allow_pickle=False)
        self.proc.stdin.write(json.dumps(dict(views=str(self.views),target=goal,plans=[r['actions'] for r in rows],arm=arm))+'\n')
        self.proc.stdin.flush()
        result=json.loads(self.proc.stdout.readline())
        if 'error' in result:raise RuntimeError(result['error'])
        return result

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=30)
        self.tmp.cleanup()


def rollout(sim, ep, spec, worker, arm):
    sim.seed(1128)
    state=hs.AgentState();state.position=np.float32(ep['start_position']);state.rotation=quat_from_coeffs(ep['start_rotation'])
    sim.get_agent(0).set_state(state)
    assert sim.pathfinder.is_navigable(state.position,max_y_delta=.08)
    obs=sim.get_sensor_observations();pose=policy_pose(sim)
    grid=DepthGrid(pose[:2]); memory=PriorEvidenceMemory()
    stats=Counter(); positions=[state.position.tolist()]
    grid.update(pose,obs['depth']); memory.mark_view(pose[:2],pose[2],visible_map_surfaces(grid,pose,1.))
    def act(i):
        nonlocal pose,obs
        obs=sim.step({1:'move_forward',2:'turn_left',3:'turn_right'}[i])
        pose=policy_pose(sim);grid.update(pose,obs['depth'])
        memory.mark_view(pose[:2],pose[2],visible_map_surfaces(grid,pose,1.))
        positions.append(sim.get_agent(0).state.position.tolist())
        stats['steps']+=1
        return obs['rgb'][:,:,:3].copy(),bool(obs.get('collided',False))
    while stats['steps']<spec['budget']:
        if spec['budget']-stats['steps']<5:
            for _ in range(spec['budget']-stats['steps']):act(2)
            break
        center=obs['rgb'][:,:,:3].copy()
        right,_=act(3); act(2); left,_=act(2); act(3)
        # Short observed-free ray prefixes plus longer translations and bent paths.
        basic=frontier_plans(grid,pose,1.,spec['budget']-stats['steps'])
        for turn in (0,-1,1,-2,2):
            rotation=[2 if turn>0 else 3]*abs(turn)
            for n in (8,3,1):
                basic.append((f"depth/{turn}/{n}",rotation+[1]*n))
        rows=candidates(grid,pose,1.,basic,spec['budget']-stats['steps'])
        if not rows:
            stats['empty_decisions']+=1
            act(2)
            continue
        predict=worker.score([right,center,left],rows,ep['object_category'],arm)
        origin=pose
        for angle,score in zip((-math.pi/6,0,math.pi/6),predict['current']):
            memory.mark_view(origin[:2],origin[2]+angle,observed_prior=score)
        for row,score,evidence in zip(rows,predict['score'],predict['evidence']):
            row['prior']=score;row['evidence']=evidence
        if arm=='new':
            index,details=memory.choose(rows); stats[details['phase']]+=1
        else:
            index=int(np.argmax(predict['score']))
        chosen=rows[index];stats['chosen']+=1
        tracker=ViewFollower(pose,1.,chosen['actions'])
        if chosen['key'].startswith('global'):stats['global_chosen']+=1
        for _ in range(min(len(chosen['actions'])+8,spec['budget']-stats['steps'])):
            # Explicit diagnostic GPS/heading is shared across all arms.
            tracker.x,tracker.y,tracker.yaw=pose
            a,arrived=tracker.next_action()
            if a is None or arrived:break
            prior_pose=pose
            prev=sim.get_agent(0).state.position.copy()
            _,collision=act(a)
            if a==1 and (collision or np.linalg.norm(sim.get_agent(0).state.position-prev)<.08):
                stats['collisions']+=1;grid.record_contact(prior_pose)
                memory.mark_blocked(chosen['position'],chosen['heading'])
                break
        close=np.linalg.norm(np.subtract(chosen['position'],pose[:2]))<=.22
        aligned=abs(math.remainder(chosen['heading']-pose[2],math.tau))<=math.radians(16)
        if close and aligned:stats['arrived']+=1
    report=measure(sim,ep,positions)
    return dict(arm=arm,scene=spec['scene'],episode=spec['id'],goal=ep['object_category'],
                steps=stats['steps'],stats=dict(stats),**report)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--indices",type=int,nargs="+",required=True)
    p.add_argument("--arms",nargs="+",choices=['old','new','prior_only'],default=['old','new'])
    args=p.parse_args()
    panel=json.loads((HERE/"panel.json").read_text())
    results=HERE/'runs';results.mkdir(exist_ok=True)
    worker=Worker()
    try:
        for i in args.indices:
            spec=panel[i];shard=DATA/'eval/hssdov/content'/(spec['scene']+'.json.gz')
            assert hashlib.sha256(shard.read_bytes()).hexdigest()==spec['shard_sha256']
            ep=next(e for e in json.load(gzip.open(shard))['episodes'] if str(e['episode_id'])==str(spec['id']))
            sim=sensor_sim(ep)
            try:
                for arm in args.arms:
                    path=results/(str(i)+'_'+arm+'.json')
                    if path.exists():continue
                    row=rollout(sim,ep,spec,worker,arm)
                    assert row['instances_available']>=2 and row['steps']==spec['budget']
                    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(path)
                    print("FINISHED",json.dumps(row),flush=True)
            finally:sim.close()
    finally:worker.close()

if __name__=="__main__":main()
