"""Generate equal-action forks in Habitat with true instance-identity, visibility and physical audits."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import habitat_sim as hs
from habitat_sim.utils.common import quat_from_angle_axis
import magnum as mn
import numpy as np

NAMES=("move_forward","turn_left","turn_right")
LOCAL=[1,1]+[0]*9+[2,2]+[0]*11
PLANS=np.array([[1,2,1,2]+LOCAL,[1]*4+LOCAL,[2]*4+LOCAL],dtype="int8")
WORDS=("cup","sofa")
# The same physical ID may be visible repeatedly along a route. Count it exactly once.
PATTERNS={
    "usual":([0,1,2],True),"swapped":([1,0,2],True),"rare_distractor":([2,0,1],True),
    "no_cue":([0,1,2],False),"usual_multi_cup":([0,1,0],True),
    "usual_multi_sofa":([0,1,1],True),"rare_multi_cup":([1,0,0],True),
    "rare_multi_sofa":([1,0,1],True),"no_cue_multi_cup":([0,1,0],False),
    "no_cue_multi_sofa":([0,1,1],False),"absent_cup":([2,1,2],True),"absent_sofa":([0,2,2],True)}


def configure(stage):
    c=hs.SimulatorConfiguration()
    c.scene_id=str(stage);c.gpu_device_id=-1;c.enable_physics=True;c.allow_sliding=False
    c.override_scene_light_defaults=True;c.scene_light_setup=hs.gfx.DEFAULT_LIGHTING_KEY
    agent=hs.agent.AgentConfiguration(height=1.45,radius=.12)
    for name,sensor in (("rgb",hs.SensorType.COLOR),("inst",hs.SensorType.SEMANTIC)):
        cam=hs.CameraSensorSpec();cam.uuid=name;cam.sensor_type=sensor
        cam.position=mn.Vector3(0,1.25,0);cam.resolution=[256,256];cam.hfov=90.
        agent.sensor_specifications.append(cam)
    agent.action_space={k:hs.agent.ActionSpec(k,hs.agent.ActuationSpec(amount=v)) for k,v in zip(NAMES,(.25,30.,30.))}
    return hs.Configuration(c,[agent])


def generate(root,house):
    folder=root/"assets"/f"tri11_{house:02}"
    split="train" if house<=8 else ("calibration" if house<=10 else "test")
    name=f"tri11_{house:02}"
    output=root/split/name;output.mkdir(exist_ok=True,parents=True)
    fingerprints={};context_groups={}; verified=0;outcomes=[];rgb_seen={};worst_return=0.;worst_length=0.
    with hs.Simulator(configure(folder/"room.stage_config.json")) as sim:
        om=sim.get_rigid_object_manager();tm=sim.get_object_template_manager()
        cats=("cup","sofa","red_chair","coffee_station","lamp")
        templates=[tm.load_configs(str(folder/(c+".object_config.json")))[0] for c in cats]
        agent=sim.get_agent(0);objects=[]
        for mode,(types,cues) in PATTERNS.items():
            for order in itertools.permutations(range(3)):
                for ob in objects:om.remove_object_by_id(ob.object_id)
                objects=[];targets=[];classes=[]
                def place(asset,side,x,distance,sem):
                    t=(0.,2*math.pi/3,-2*math.pi/3)[side];co,si=np.cos(t),np.sin(t)
                    obj=om.add_object_by_template_id(templates[asset])
                    obj.translation=mn.Vector3(co*x-si*distance,0.,-(si*x+co*distance))
                    obj.motion_type=hs.physics.MotionType.STATIC;obj.semantic_id=sem;objects.append(obj)
                    return obj
                for side,rank in enumerate(order):
                    cat=types[rank];classes.append(cat);targets.append(place(cat,side,-2.03,5.4,101+side))
                    if cues and rank<2:place(3+rank,side,.58,1.4,301+side)
                nav=hs.NavMeshSettings();nav.set_defaults()
                nav.agent_height,nav.agent_radius,nav.include_static_objects=1.45,.12,True
                assert sim.recompute_navmesh(sim.pathfinder,nav)
                origin=np.array(sim.pathfinder.snap_point(np.zeros(3,dtype=np.float32)))
                def reset():
                    s=hs.AgentState();s.position=origin;s.rotation=quat_from_angle_axis(0.,np.array([0.,1.,0.]));agent.set_state(s)
                reset();ctx=[];cue_pix=[];leak=0
                for i in range(12):
                    obs=sim.get_sensor_observations();ctx.append(obs["rgb"][:,:,:3].copy())
                    leak=max(leak,int(np.isin(obs["inst"],[101,102,103]).sum()))
                    cue_pix.append([np.count_nonzero(obs["inst"]==301+s) for s in range(3)])
                    agent.act("turn_left")
                assert not leak,(name, mode, order, leak)
                ctx=np.array(ctx);cue_pix=np.array(cue_pix)
                assert (cue_pix.max(axis=0)>=64).sum()==(2 if cues else 0)
                group=(bool(cues), order if cues else ())
                hx=hashlib.sha256(ctx.tobytes()).hexdigest()
                assert context_groups.setdefault(group,hx)==hx,(group,mode)
                context_groups[group]=hx
                future=[];collisions=[];ids=[];dist=[];seen=[]
                for j,plan in enumerate(PLANS):
                    reset();frames=[];hits=[];gains=set();dis=0.;mask_peak=np.zeros(3,int)
                    for step,action in enumerate(plan,1):
                        prior=agent.get_state().position.copy();hits.append(bool(agent.act(NAMES[action])))
                        pos=agent.get_state().position.copy();dis+=float(np.linalg.norm((pos-prior)[[0,2]]))
                        ob=sim.get_sensor_observations()
                        px=np.array([np.count_nonzero(ob["inst"]==101+i) for i in range(3)])
                        close=np.array([np.linalg.norm((np.array(t.translation)-pos)[[0,2]])<=2. for t in targets])
                        mask_peak=np.maximum(mask_peak,px*close)
                        gains.update(np.flatnonzero(close & (px>=64)).tolist())
                        if step in (7,14,21,28):frames.append(ob["rgb"][:,:,:3].copy())
                    assert gains=={j} and px[j]>=64 and close[j],(name,mode,j,gains,px)
                    assert abs(dis-5.)<.015,(name,mode,j,dis)
                    for a in [1]*6+[{0:0,1:2,2:1}[v] for v in reversed(plan)]+[1]*6:agent.act(NAMES[a])
                    error=float(np.linalg.norm(agent.get_state().position-origin))
                    assert error<.035,(name,error)
                    worst_return=max(worst_return,error);worst_length=max(worst_length,abs(dis-5.))
                    future.append(frames);collisions.append(hits);dist.append(dis)
                    ids.append([sorted(101+i for i in gains if classes[i]==q) for q in range(len(WORDS))])
                    seen.append(mask_peak)
                y=np.array([[len(v) for v in u] for u in ids],np.int8)
                identity=np.full((3,len(WORDS),3),-1,np.int16)
                for i,group_ids in enumerate(ids):
                    for q,instance_ids in enumerate(group_ids):identity[i,q,:len(instance_ids)]=instance_ids
                oid=hashlib.sha256(f"{name}:{order}:{mode}".encode()).hexdigest()[:18]
                # Only RGB, headings and candidate actions are policy-side inputs prior to dispatch.
                np.savez_compressed(output/(oid+".npz"),ctx=ctx,headings=np.arange(12)*30.,actions=PLANS,
                    future=np.array(future),future_steps=[7,14,21,28],words=np.array(WORDS),y=y,
                    house=name,origin=oid,source="blender",group=mode,collided=np.array(collisions),
                    hit_instance_ids=identity,initial_target_pixels=leak,cue_pixels=cue_pix,
                    near_instance_pixels=np.array(seen),branch_metres=dist)
                outcomes.append(dict(mode=mode,origin=oid,counts=y.sum(axis=0).tolist(),cues_visible_input=int((cue_pix[[11,0,1]].max(0)>=64).sum())))
                verified+=1
    receipt=dict(house=name,split=split,starts=verified,matched_visual_groups=len(context_groups),
        hidden_start_goal_pixels=0,max_distance_error_m=worst_length,max_return_error_m=worst_return,
        unique_entrance_sha=len(set(context_groups.values())),layouts=outcomes)
    (output/"receipt.json").write_text(json.dumps(receipt))
    print("verified",name,verified,flush=True)


if __name__=="__main__":generate(Path(sys.argv[1]).resolve(),int(sys.argv[2]))
