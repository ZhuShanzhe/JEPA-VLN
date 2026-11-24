"""Execute side scans, contact probes and door changes inside one Habitat instance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import math
import habitat_sim as hs
import magnum as mn
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis

D=Path(__file__).resolve().parent
def config(stage):
    sim=hs.SimulatorConfiguration();sim.scene_id=str(stage);sim.gpu_device_id=-1;sim.allow_sliding=False
    sim.enable_physics=True;sim.override_scene_light_defaults=True;sim.scene_light_setup=hs.gfx.DEFAULT_LIGHTING_KEY
    ag=hs.agent.AgentConfiguration(height=1.42,radius=.12)
    for n,t in [('rgb',hs.SensorType.COLOR),('semantic',hs.SensorType.SEMANTIC)]:
        sp=hs.CameraSensorSpec();sp.uuid=n;sp.sensor_type=t;sp.resolution=[128,128];sp.hfov=85.;sp.position=mn.Vector3(0,1.18,0)
        ag.sensor_specifications.append(sp)
    ag.action_space={k:hs.agent.ActionSpec(k,hs.agent.ActuationSpec(amount=a)) for k,a in
        [('move_forward',.25),('turn_left',30.),('turn_right',30.)]}
    return hs.Configuration(sim,[ag])

def go(spec,initial,swapped):
    assets=D/'assets';name=spec['id'];uid=f'{name}_{initial}_{swapped}'
    out=D/'records_v3';out.mkdir(exist_ok=True)
    if (out/(uid+'.truth.json')).exists():return
    with hs.Simulator(config(assets/(name+'.stage_config.json'))) as sim:
        templates=sim.get_object_template_manager();objects=sim.get_rigid_object_manager()
        def spawn(kind,where,sid):
            t=templates.load_configs(str(assets/(kind+'.object_config.json')))[0]
            ob=objects.add_object_by_template_id(t);ob.translation=mn.Vector3(*where)
            ob.motion_type=hs.physics.MotionType.STATIC;ob.semantic_id=sid;return ob
        gatepos=lambda i,closed:[(-1,1)[i]*spec['dist'][i],0.,0. if closed else 8.]
        gates=[spawn(name+'_gate',gatepos(i,i==initial),50+i) for i in range(2)]
        targets=[np.array([(-1,1)[i]*spec['near_target_x'][i],0.,spec['near_target_z']]) for i in range(2)]
        for i,p in enumerate(targets):spawn('small_goal',p,101+i)
        # Both changed and stable-door episodes contain a real, nonblocking visual change.
        # The same geometry occupies a side strip; it cannot obstruct the navigation line.
        def marker(i,side):return [(-1,1)[i]*.92,.18,side*spec['width']*.32]
        markers=[spawn('small_goal',marker(i,-1),121+i) for i in range(2)]
        settings=hs.NavMeshSettings();settings.set_defaults();settings.agent_radius=.12;settings.agent_height=1.42;settings.include_static_objects=True
        def bake():
            if not sim.recompute_navmesh(sim.pathfinder,settings):raise RuntimeError('navmesh build')
        bake()
        actor=sim.get_agent(0);st=hs.AgentState();st.position=sim.pathfinder.snap_point([0.,0.,0.])
        st.rotation=quat_from_angle_axis(0.,np.array([0.,1.,0.]));actor.set_state(st);origin=actor.get_state().position.copy()
        rgb=lambda:sim.get_sensor_observations()['rgb'][:,:,:3].copy()
        def align(which):
            for _ in range(3):actor.act('turn_left' if which==0 else 'turn_right')
        def scan():
            frames=[]
            for i in range(2):
                align(i);frames.append(rgb());align(1-i)
            return np.asarray(frames)
        def probe(i):
            align(i);start=actor.get_state().position.copy();observed_block=False;distances=[];motion_rgb=[]
            dist4=dist12=0.;near_hit=False;near_pixels=0
            for step in range(1,13):
                before=rgb();prev=actor.get_state().position.copy();actor.act('move_forward')
                now=actor.get_state().position.copy();o=sim.get_sensor_observations();after=o['rgb'][:,:,:3]
                # Only RGB-derived absence of ego-motion is made available as a stale observation.
                moved_rgb=bool(np.abs(after.astype(np.float32)-before).mean()>=.80)
                observed_block|=not moved_rgb;motion_rgb.append(moved_rgb)
                moved=float(np.linalg.norm(now-prev))
                distance=float(np.linalg.norm(now-start))
                if moved>1e-5:distances.append(moved)
                if step==4:dist4=distance
                dist12=distance
                pix=int(np.sum(o['semantic']==101+i))
                near_pixels=max(near_pixels,pix)
                if step<=4 and pix>=12 and np.linalg.norm((now-targets[i])[[0,2]])<1.:near_hit=True
            # The last motion can be partial. Reverse its measured physical length as
            # part of the collector return controller, keeping candidate plans at .25 m.
            for _ in range(6):actor.act('turn_left')
            command=actor.agent_config.action_space['move_forward'].actuation
            for moved in reversed(distances):
                command.amount=moved;actor.act('move_forward')
            command.amount=.25
            for _ in range(6):actor.act('turn_left')
            align(1-i)
            drift=float(np.linalg.norm(actor.get_state().position-origin))
            if drift>.018:raise RuntimeError(f'return drift {name}/{i}: {drift:.3f}')
            return dict(blocked_rgb=observed_block,motion_rgb=motion_rgb,d4=dist4,d12=dist12,near=near_hit,near_pixels=near_pixels,drift=drift)
        before=scan();first=[probe(i) for i in range(2)]
        neutral=rgb()
        if swapped:
            for i,g in enumerate(gates):
                g.motion_type=hs.physics.MotionType.KINEMATIC;g.translation=mn.Vector3(*gatepos(i,i!=initial));g.motion_type=hs.physics.MotionType.STATIC
        for i,m in enumerate(markers):
            m.motion_type=hs.physics.MotionType.KINEMATIC;m.translation=mn.Vector3(*marker(i,1));m.motion_type=hs.physics.MotionType.STATIC
        bake()
        event_visible=float(np.abs(neutral.astype(float)-rgb()).mean())
        after=scan();second=[probe(i) for i in range(2)]
        obs=out/(uid+'.obs.npz');np.savez_compressed(obs,before=before,after=after,prior_blocked=[x['blocked_rgb'] for x in first],prior_movement=[x['motion_rgb'] for x in first])
        truth=dict(id=name,initial=initial,swapped=swapped,neutral_event_rgb_mae=event_visible,
                   outcomes=[[{k:v for k,v in s.items() if k not in ('blocked_rgb','motion_rgb')} for s in phase] for phase in (first,second)])
        (out/(uid+'.truth.json')).write_text(json.dumps(truth))
if __name__=='__main__':
    import sys
    name=sys.argv[1];spec=next(r for r in json.loads((D/'assets/spec.json').read_text()) if r['id']==name)
    for i in range(2):
        for sw in (0,1):go(spec,i,sw)
    print('complete',name,flush=True)
