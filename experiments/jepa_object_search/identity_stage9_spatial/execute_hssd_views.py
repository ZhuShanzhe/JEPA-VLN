"""Execute reachable discrete view choices from held-out HSSD registration poses.

The starting detection box is supplied by the archived annotation. Only RGB,
that box, its current depth and navmesh enter the predictors. Object labels are
used after the choice to measure how much of that particular object was seen.
"""
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys

import habitat_sim as hs
from habitat_sim.utils.common import quat_from_angle_axis
import magnum as mn
import numpy as np
import quaternion

from view_jepa import ViewPredictor

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'jepa_reid_v20/collect'))
from collect_views import inventory


def simulator(house):
    cfg=hs.SimulatorConfiguration()
    cfg.scene_dataset_config_file=str(ROOT/'simulation/hssd-render/hssd-hab.scene_dataset_config.json')
    cfg.scene_id=house
    cfg.enable_physics=True
    ag=hs.agent.AgentConfiguration(height=1.4,radius=.18)
    for label,typ in [('rgb',hs.SensorType.COLOR),('depth',hs.SensorType.DEPTH),('ids',hs.SensorType.SEMANTIC)]:
        s=hs.CameraSensorSpec()
        s.uuid,s.sensor_type,s.resolution=label,typ,[384,384]
        s.position,s.hfov=mn.Vector3(0,1.25,0),mn.Deg(75)
        ag.sensor_specifications.append(s)
    for turn in ['turn_left','turn_right']:
        ag.action_space[turn].actuation.amount=30.
    ag.action_space['move_forward'].actuation.amount=.25
    sim=hs.Simulator(hs.Configuration(cfg,[ag]))
    nav=hs.NavMeshSettings();nav.set_defaults()
    nav.agent_height,nav.agent_radius,nav.include_static_objects=1.4,.18,True
    assert sim.recompute_navmesh(sim.pathfinder,nav)
    inventory(sim,ROOT/'simulation/hssd-render')
    return sim


def turns(n):
    return ['turn_left' if n>0 else 'turn_right']*abs(n)


def measure(mask,instance):
    yy,xx=np.where(mask==instance)
    if xx.size==0:
        return dict(visible_pixels=0,short_side_px=0,fully_in_frame=False)
    return dict(visible_pixels=xx.size,short_side_px=int(min(xx.ptp()+1,yy.ptp()+1)),
                fully_in_frame=bool(xx.min()>1 and yy.min()>1 and xx.max()<382 and yy.max()<382))


def execute(sim,r,models):
    agent=sim.get_agent(0)
    eye=np.array(r['camera_xyz'])
    pos=eye-[0,1.25,0]
    heading=math.radians(r['camera_heading_deg'])
    yaw=math.atan2(-math.cos(heading),-math.sin(heading))
    state=hs.AgentState()
    state.position=pos
    state.rotation=quat_from_angle_axis(yaw,np.array([0.,1.,0.]))
    agent.set_state(state)
    initial=agent.get_state()
    orientation=initial.rotation*quat_from_angle_axis(math.radians(r['pitch_deg']),np.array([1.,0.,0.]))
    for s in initial.sensor_states.values():
        s.rotation=orientation
    agent.set_state(initial,infer_sensor_states=False,reset_sensors=False)
    start=sim.get_sensor_observations()
    sid=int(r['instance'].split(':')[-1])
    original=measure(start['ids'],sid)
    if original['visible_pixels']!=r['pixels']:
        return dict(status='start_did_not_reproduce',source=original,archived_pixels=r['pixels'])
    box=np.asarray(r['bbox']); u,v=(box[:2]+box[2:])*.5
    z=np.median(start['depth'][int(v)-2:int(v)+3,int(u)-2:int(u)+3])
    if not np.isfinite(z) or z<=0:return dict(status='missing_registered_depth')
    focal=192./math.tan(math.radians(37.5))
    target=(eye+quaternion.rotate_vectors(orientation,np.array([(u-191.5)*z/focal,(191.5-v)*z/focal,-z])))[[0,2]]
    origin=pos[[0,2]]-target
    options=[]
    for nturn in (-2,-1,0,1,2):
        ang=yaw+math.radians(30)*nturn
        delta=.25*np.array([-math.sin(ang),0.,-math.cos(ang)])
        end=pos.copy()
        for nforward in (1,2,3):
            proposed=end+delta
            end=np.array(sim.pathfinder.try_step(end,proposed))
            if not np.isfinite(end).all() or np.linalg.norm((end-proposed)[[0,2]])>.023:break
            direction=target-end[[0,2]]
            if np.linalg.norm(direction)<.4:continue
            orient=math.atan2(-direction[0],-direction[1])
            finish=round(math.remainder(orient-ang,math.tau)/math.radians(30))
            actions=turns(nturn)+['move_forward']*nforward+turns(finish)
            offset=end[[0,2]]-target
            angle=math.remainder(math.atan2(offset[1],offset[0])-math.atan2(origin[1],origin[0]),math.tau)
            options.append((actions,end,angle,np.linalg.norm(offset)/np.linalg.norm(origin)))
    if len(options)<2:return dict(status='too_few_reachable_views',source=original,n=len(options))
    picks=[int(np.argmax(m.score_views(start['rgb'][:,:,:3],box,[q[2] for q in options],[q[3] for q in options]))) for m in models]
    outcomes=[]
    for actions,endpoint,_,_ in options:
        agent.set_state(initial,infer_sensor_states=False,reset_sensors=False)
        f=None
        for action in actions:
            f=sim.step(action)
        end=np.array(agent.get_state().position)
        error=np.linalg.norm((end-endpoint)[[0,2]])
        assert error<.04,(actions,error)
        outcomes.append(dict(**measure(f['ids'],sid),actions=len(actions),moved_m=round(float(np.linalg.norm((end-pos)[[0,2]])),3),
                             endpoint_error_m=round(float(error),5)))
    return dict(status='executed',source=original,n=len(options),same_choice=picks[0]==picks[1],
                jepa=outcomes[picks[0]],direct=outcomes[picks[1]],
                mean_reachable_pixels=round(float(np.mean([o['visible_pixels'] for o in outcomes])),1),
                max_reachable_pixels=max(o['visible_pixels'] for o in outcomes))


def main():
    models=[ViewPredictor(HERE/'weights'/(name+'.npz')) for name in ('view_jepa','direct')]
    jobs=defaultdict(list)
    for folder,cat in [('clocks','clock'),('data','wall_art')]:
        rows=[json.loads(l) for l in (ROOT/'jepa_reid_v20'/folder/'manifest.jsonl').read_bytes().splitlines()]
        # First two SOURCE poses of the lexicographically first eligible instance.
        eligible=sorted((r for r in rows if r['split']=='test' and r['category']==cat),
                        key=lambda r:(r['house'],r['instance'],r['view']))
        if eligible:
            first=eligible[0]['instance']
            for r in [r for r in eligible if r['instance']==first][:2]:jobs[r['house']].append(r)
    summary=[]
    for house,rows in jobs.items():
        sim=simulator(house)
        try:
            for r in rows:
                result=dict(category=r['category'],source_view=r['view'],**execute(sim,r,models))
                print(json.dumps(result),flush=True)
                summary.append(result)
        finally:
            sim.close()
    output=dict(protocol='Four preordered held-out registrations with supplied source boxes, measured start depth, navmesh reachable translates, deterministic physical execution and post-choice target labels.',outcomes=summary)
    (HERE/'hssd_physical_views.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
