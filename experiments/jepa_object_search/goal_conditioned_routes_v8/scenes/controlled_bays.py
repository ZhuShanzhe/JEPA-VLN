"""Measure target-dependent route reversals in a user-supplied, two-branch Habitat scene."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def observe_swap(spec, root, mode, swapped, device):
    import habitat_sim as hs
    import magnum as mn
    from habitat_sim.utils.common import quat_from_angle_axis

    sim_cfg = hs.SimulatorConfiguration()
    sim_cfg.scene_id = str((root / spec["stage"]).resolve())
    sim_cfg.gpu_device_id = device
    sim_cfg.enable_physics = True
    sim_cfg.allow_sliding = False
    sim_cfg.override_scene_light_defaults = True
    sim_cfg.scene_light_setup = hs.gfx.DEFAULT_LIGHTING_KEY
    ac = hs.agent.AgentConfiguration(height=1.45, radius=.12)
    sensors = []
    for name, kind in (("rgb",hs.SensorType.COLOR),("semantic",hs.SensorType.SEMANTIC)):
        sensor = hs.CameraSensorSpec()
        sensor.uuid, sensor.sensor_type = name, kind
        sensor.position, sensor.hfov = mn.Vector3(0,1.25,0), 90
        sensor.resolution = [256,256]
        sensors.append(sensor)
    ac.sensor_specifications = sensors
    actions = ["move_forward","turn_left","turn_right"]
    ac.action_space = {k:hs.agent.ActionSpec(k,hs.agent.ActuationSpec(amount=v)) for k,v in zip(actions,(.25,30.,30.))}
    targets = spec["targets"]
    routes = spec["action_ids"]
    if len(targets)!=2 or len(routes)!=2 or any(len(a)!=24 or set(a)-{0,1,2} for a in routes):
        raise ValueError("This diagnostic requires two goal classes and two fixed 24-action routes.")
    if (routes[0].count(0),len(set(t["query"] for t in targets))) != (routes[1].count(0),2):
        raise ValueError("Routes must have equal forward counts and targets need distinct query names.")
    with hs.Simulator(hs.Configuration(sim_cfg,[ac])) as sim:
        tm=sim.get_object_template_manager()
        om=sim.get_rigid_object_manager()
        def place(template, xyz, semantic):
            ids=tm.load_configs(str((root/template).resolve()))
            if not ids:
                raise ValueError(f"Unresolved supplied object template {template}")
            obj=om.add_object_by_template_id(ids[0])
            obj.translation=mn.Vector3(*xyz)
            obj.motion_type=hs.physics.MotionType.STATIC
            obj.semantic_id=semantic
        order=[1,0] if swapped else [0,1]
        for side,index in enumerate(order):
            place(targets[index]["object_template"],spec["target_positions"][side],101+side)
            if mode!="hidden":
                cue=index if mode=="congruent" else 1-index
                place(targets[cue]["cue_template"],spec["cue_positions"][side],301+side)
        nav=hs.NavMeshSettings()
        nav.set_defaults()
        nav.agent_height,nav.agent_radius,nav.include_static_objects=1.45,.12,True
        if not sim.recompute_navmesh(sim.pathfinder,nav):
            raise ValueError("No navigable mesh in supplied fixture")
        pf=sim.pathfinder
        agent=sim.get_agent(0)
        authored=np.array(spec["start"],np.float32)
        start=np.array(pf.snap_point(authored))
        if not np.isfinite(start).all() or np.linalg.norm((start-authored)[[0,2]])>.05:
            raise ValueError("Unreachable supplied starting location")
        def put_camera(pos,yaw):
            state=hs.AgentState()
            state.position=np.asarray(pos,np.float32)
            state.rotation=quat_from_angle_axis(float(yaw),np.array([0.,1.,0.]))
            agent.set_state(state)
        # Independent of both candidate trajectories. Check physical navigation and target pixels.
        catalog=[]
        for side,center in enumerate(spec["target_positions"]):
            approved=[]
            for dx in np.arange(-1.75,1.751,.25):
                for dz in np.arange(-1.75,1.751,.25):
                    if not .55<=np.hypot(dx,dz)<=1.85:
                        continue
                    requested=np.array([center[0]+dx,start[1],center[2]+dz],np.float32)
                    point=np.asarray(pf.snap_point(requested))
                    if not np.isfinite(point).all() or np.linalg.norm(point-requested)>.11:
                        continue
                    put_camera(point,np.arctan2(point[0]-center[0],point[2]-center[2]))
                    mask=sim.get_sensor_observations()["semantic"]==101+side
                    if np.count_nonzero(mask)>=64:
                        approved.append(point.tolist())
            catalog.append(np.unique(np.asarray(approved,dtype=np.float32).reshape(-1,3),axis=0))
        def eligible(pos):
            found=np.zeros(2,bool)
            for side,points in enumerate(catalog):
                close=points[np.linalg.norm(points-pos,axis=1)<=.25001]
                if len(close):
                    path=hs.MultiGoalShortestPath()
                    path.requested_start=np.asarray(pos,np.float32)
                    path.requested_ends=close.tolist()
                    found[side]=bool(pf.find_path(path) and path.geodesic_distance<=.25)
            return found
        yaw=np.deg2rad(spec.get("start_yaw_deg",0.))
        put_camera(start,yaw)
        digest=hashlib.sha256()
        target_pixels=np.zeros(2,int)
        cue_pixels=np.zeros(2,int)
        for _ in range(12):
            obs=sim.get_sensor_observations()
            digest.update(np.ascontiguousarray(obs["rgb"][:,:,:3]).tobytes())
            for side in range(2):
                target_pixels[order[side]]=max(target_pixels[order[side]],np.count_nonzero(obs["semantic"]==101+side))
                cue_pixels[side]=max(cue_pixels[side],np.count_nonzero(obs["semantic"]==301+side))
            agent.act("turn_left")
        excluded=eligible(start)
        gains={str(h):{t["query"]:[] for t in targets} for h in (4,12,24)}
        total=[]
        for commands in routes:
            put_camera(start,yaw)
            seen=np.zeros(2,bool)
            distance=0.
            for step,action in enumerate(commands,1):
                before=agent.get_state().position
                agent.act(actions[action])
                after=agent.get_state().position
                distance+=float(np.linalg.norm(after-before))
                seen|=eligible(after)&~excluded
                if str(step) in gains:
                    for side,idx in enumerate(order):
                        gains[str(step)][targets[idx]["query"]].append(int(seen[side]))
            total.append(distance)
        return {"mode":mode,"swapped":swapped,"context_rgb_sha256":digest.hexdigest(),"maximum_target_pixels":target_pixels.tolist(),
                "maximum_cue_pixels":cue_pixels.tolist(),"independent_view_count":[len(v) for v in catalog],
                "route_lengths_m":total,"target_gain":gains}


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--device",type=int,default=0)
    args=ap.parse_args()
    cfg=json.loads(args.scene.read_text(encoding="utf-8"))
    records=[observe_swap(cfg,args.scene.parent,mode,flag,args.device)
             for mode in ("congruent","hidden","conflicting") for flag in (False,True)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as f:json.dump(records,f,ensure_ascii=False,indent=2)
