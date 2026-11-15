"""Render paired routes and isolated teacher labels in the exported Habitat layouts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import os
import numpy as np
import magnum as mn
import habitat_sim as hs


def config(stage):
    cfg=hs.SimulatorConfiguration()
    cfg.scene_id=str(stage)
    cfg.gpu_device_id=-1
    cfg.enable_physics=True
    cfg.override_scene_light_defaults=True
    cfg.scene_light_setup=hs.gfx.DEFAULT_LIGHTING_KEY
    agent=hs.agent.AgentConfiguration(height=1.45,radius=.12)
    for name,kind in [("rgb",hs.SensorType.COLOR),("labels",hs.SensorType.SEMANTIC)]:
        cam=hs.CameraSensorSpec()
        cam.uuid,cam.sensor_type,cam.position,cam.resolution,cam.hfov=name,kind,mn.Vector3(0,1.15,0),[192,192],68
        agent.sensor_specifications.append(cam)
    agent.action_space={k:hs.agent.ActionSpec(k,hs.agent.ActuationSpec(amount=v)) for k,v in
                        (("move_forward",.25),("turn_left",15.),("turn_right",15.))}
    return hs.Configuration(cfg,[agent])


def capture(assets,house,items,cues=None):
    with hs.Simulator(config(assets/(house+".stage_config.json"))) as sim:
        tm=sim.get_object_template_manager()
        om=sim.get_rigid_object_manager()
        for i,(sign,item) in enumerate(zip((-1,1),items)):
            ids=tm.load_configs(str(assets/(item+".object_config.json")))
            obj=om.add_object_by_template_id(ids[0])
            obj.translation=mn.Vector3(sign*2.03,0,-1.65)
            obj.motion_type=hs.physics.MotionType.STATIC
            obj.semantic_id=101+i
        if cues is not None:
            for i,(sign,key) in enumerate(zip((-1,1),cues)):
                template=tm.load_configs(str(assets/(key+".object_config.json")))
                cue=om.add_object_by_template_id(template[0])
                cue.translation=mn.Vector3(sign*.78,0,2.10)
                cue.motion_type=hs.physics.MotionType.STATIC
                cue.semantic_id=301+i
        nav=hs.NavMeshSettings()
        nav.set_defaults()
        nav.agent_height,nav.agent_radius,nav.include_static_objects=1.45,.12,True
        if not sim.recompute_navmesh(sim.pathfinder,nav):
            raise ValueError("No traversable mesh")
        start=sim.pathfinder.snap_point(np.array([0,0,3.75],dtype=np.float32))
        if not np.isfinite(start).all() or abs(start[2]-3.75)>.05:
            raise ValueError("Invalid start")
        futures,area,actions,xy,travel=[],[],[],[],[]
        initial=None
        for side in (0,1):
            state=hs.AgentState()
            state.position=start
            sim.get_agent(0).set_state(state)
            codes=[1+side]*4+[0]*9+[2-side]*4+[0]*12
            history,counts,positions=[],[],[]
            for a in [-1]+codes:
                frame=sim.get_sensor_observations() if a==-1 else sim.step(("move_forward","turn_left","turn_right")[a])
                if frame.get("collided",False):
                    raise ValueError(f"Collision {house} {side} {len(history)}")
                history.append(frame["rgb"][:,:,:3].copy())
                counts.append([(frame["labels"]==v).sum() for v in (101,102)])
                positions.append(sim.get_agent(0).state.position.copy())
                if a==-1 and cues is not None and any((frame["labels"]==v).sum()<38 for v in (301,302)):
                    raise ValueError("The branch cues must be visible before the first action")
            if initial is not None and not np.array_equal(initial,history[0]):
                raise ValueError("Unequal initial images")
            initial=history[0]
            futures.append(np.stack(history[1:]))
            area.append(counts)
            actions.append(codes)
            # Only nominal control geometry enters the model.
            target=[3+2.25*math.cos(math.pi/3), (1 if side==0 else -1)*2.25*math.sin(math.pi/3)]
            xy.append(target)
            expected=np.array([-target[1],0,-target[0]])
            measured=positions[-1]-positions[0]
            if np.linalg.norm((measured-expected)[[0,2]])>.05:
                raise ValueError("Route did not physically execute")
            travel.append(float(np.linalg.norm(measured)))
        return initial[None],np.stack(futures),np.array(area),np.array(actions),np.asarray(xy),travel


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--assets",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--cues",action="store_true",help="Use observable room cues with counterbalanced sides")
    args=p.parse_args()
    if (args.output/"index.jsonl").exists():
        raise FileExistsError("Dataset already finalized")
    args.output.mkdir(exist_ok=True,parents=True)
    records,receipts=[],[]
    for layout in json.loads((args.assets/"layouts.json").read_text()):
        house=layout["house"]
        for pair,items in enumerate([("red_chair","blue_chair"),("sofa","cup")]):
            if args.cues and pair==0:
                continue
            initials=[]
            for permutation in (0,1):
                selected=items if not permutation else tuple(reversed(items))
                cues=tuple("lamp" if item=="sofa" else "coffee_station" for item in selected) if args.cues else None
                context,future,pixels,actions,xy,travel=capture(args.assets.resolve(),house,selected,cues)
                initials.append(context)
                for j,item in enumerate(selected):
                    # Stronger sustained view than at the common start. Labels never enter context.
                    count=pixels[:,:,j]
                    sustained=np.minimum(count[:,1:-1],count[:,2:]).max(axis=1)
                    gain=((sustained >= 38) & (count[:,0] < 38)).astype(np.float32)
                    id_=f"{'cue' if args.cues else 'bay'}_{house}_{pair}_{permutation}_{item}"
                    file=id_+".npz"
                    presence=(np.arange(2)==j).astype(np.float32)
                    np.savez_compressed(args.output/file,context_rgb=context,future_rgb=future,xy=xy.astype(np.float32),
                                        actions=actions.astype(np.int64),gain=gain,region_presence=presence,
                                        step_m=np.float32(.25),turn_deg=np.float32(15))
                    query=item.replace("_"," ")
                    records.append(dict(id=id_,source="blender_bays",house=house,split=layout["split"],
                                        query=query,label_kind="contextual_exposure" if args.cues else "new_sustained_exposure",path=file))
                    receipts.append(dict(id=id_,first_pixels=count[:,0].tolist(),sustained_peak=sustained.tolist(),
                                         gain=gain.tolist(),displacement=travel))
            if args.cues and np.array_equal(*initials):
                raise ValueError("Opposite outcomes with no observable cue change")
        print("captured",house,flush=True)
    diverse={split:sum(max(r["gain"])>min(r["gain"]) for row,r in zip(records,receipts)
                       if row["split"]==split) for split in ("train","dev")}
    if min(diverse.values())==0:
        raise ValueError("No distinguishable decisions in one of the splits")
    contents="".join(json.dumps(row)+"\n" for row in records)
    (args.output/"capture_audit.json").write_text(json.dumps(dict(rows=receipts,distinguishable=diverse,
        resolution=192,threshold_pixels=38,actions_per_branch=29,arms=2,
        index_sha256=hashlib.sha256(contents.encode()).hexdigest()),indent=2))
    temp=args.output/"index.jsonl.tmp"
    temp.write_text(contents)
    os.replace(temp,args.output/"index.jsonl")
    print("accepted",len(records),diverse)


if __name__=="__main__":
    main()
