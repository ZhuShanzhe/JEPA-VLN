"""Unproject a detected box's median depth after verifying alignment against its saved RGB."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


SIZE=384
HFOV=75
HEIGHT=1.25
FOCAL=.5*SIZE/math.tan(math.radians(.5*HFOV))


def open_sim(assets,house):
    import habitat_sim as hs
    import magnum as mn
    cfg=hs.SimulatorConfiguration()
    cfg.scene_dataset_config_file=str(assets/"hssd-hab.scene_dataset_config.json")
    cfg.scene_id=house
    cfg.enable_physics=True
    agent=hs.agent.AgentConfiguration(height=1.4,radius=.18)
    for name,typ in (("rgb",hs.SensorType.COLOR),("depth",hs.SensorType.DEPTH)):
        spec=hs.CameraSensorSpec()
        spec.uuid,spec.sensor_type,spec.resolution=name,typ,[SIZE,SIZE]
        spec.position,spec.hfov=mn.Vector3(0,HEIGHT,0),mn.Deg(HFOV)
        agent.sensor_specifications.append(spec)
    return hs.Simulator(hs.Configuration(cfg,[agent]))


def observe(sim,row):
    import habitat_sim as hs
    from habitat_sim.utils.common import quat_from_angle_axis
    heading=math.radians(row["camera_heading_deg"])
    yaw=math.atan2(-math.cos(heading),-math.sin(heading))
    pitch=math.radians(row["pitch_deg"])
    state=hs.AgentState()
    state.position=np.array(row["camera_xyz"])-[0,HEIGHT,0]
    state.rotation=quat_from_angle_axis(yaw,np.array([0.,1.,0.]))
    agent=sim.get_agent(0)
    agent.set_state(state)
    state=agent.get_state()
    orient=state.rotation*quat_from_angle_axis(pitch,np.array([1.,0.,0.]))
    for sensor in state.sensor_states.values():
        sensor.rotation=orient
    agent.set_state(state,infer_sensor_states=False,reset_sensors=False)
    obs=sim.get_sensor_observations()
    return obs,agent.get_state().sensor_states["depth"]


def rgb_error(render,row,data):
    # Reproduce both the crop and the JPEG encoder used to write the reference.
    image=Image.fromarray(render[:,:,:3]).crop(row["tight_bbox"])
    buffer=BytesIO()
    image.save(buffer,format="JPEG",quality=93)
    buffer.seek(0)
    with Image.open(buffer) as recoded, Image.open(data/row["image"]) as saved:
        err=np.abs(np.asarray(recoded,dtype=np.int16)-np.asarray(saved,dtype=np.int16))
    return float(err.mean()),float((err.max(-1)>8).mean())


def estimate(depth,box,state):
    from habitat_sim.utils.common import quat_rotate_vector
    x0,y0,x1,y1=box
    crop=depth[y0:y1,x0:x1]
    finite=np.isfinite(crop)&(crop>0)&(crop<100.)
    if finite.sum()<9:
        return np.full(3,np.nan),float("nan")
    median=np.median(crop[finite])
    support=finite&(np.abs(crop-median)<=max(.03,.03*median))
    v,u=np.where(support)
    if not len(u):
        return np.full(3,np.nan),float("nan")
    mid=np.argmin((u-np.median(u))**2+(v-np.median(v))**2)
    z=float(crop[v[mid],u[mid]])
    pixel_x=x0+u[mid]+.5
    pixel_y=y0+v[mid]+.5
    point=z*np.array([(pixel_x-SIZE*.5)/FOCAL,(SIZE*.5-pixel_y)/FOCAL,-1.])
    world=np.asarray(state.position)+quat_rotate_vector(state.rotation,point)
    return np.asarray(world,dtype=np.float32),z


def per_house(a,rows):
    sim=open_sim(a.assets,a.house)
    indices,xyz,valid,mae,tail,z=[],[],[],[],[],[]
    try:
        for i,row in rows:
            obs,state=observe(sim,row)
            l1,frac=rgb_error(obs["rgb"],row,a.manifest.parent)
            aligned=(l1<=1.) and (frac<=.02)
            point,depth=estimate(obs["depth"],row["bbox"],state) if aligned else (np.full(3,np.nan),float("nan"))
            indices.append(i)
            xyz.append(point)
            valid.append(aligned and np.isfinite(point).all())
            mae.append(l1)
            tail.append(frac)
            z.append(depth)
    finally:
        sim.close()
    np.savez_compressed(a.out/"depth_parts"/(a.house+".npz"),ix=indices,xyz=xyz,valid=valid,rgb_mae=mae,rgb_bad_pixel_fraction=tail,depth=z)


def main(a):
    with a.manifest.open() as f:
        selected=[(i,r) for i,line in enumerate(f) if (r:=json.loads(line))["split"] in ("val","test")]
    groups=defaultdict(list)
    for i,row in selected:
        groups[row["house"]].append((i,row))
    if a.house is not None:
        per_house(a,groups[a.house])
        return
    parts=a.out/"depth_parts"
    parts.mkdir(parents=True,exist_ok=True)
    def one(h):
        with (parts/(h+".log")).open("w") as f:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),str(a.assets),str(a.manifest),str(a.out),"--house",h],check=True,stdout=f,stderr=f)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(one,groups))
    field_arrays=defaultdict(list)
    for h in groups:
        with np.load(parts/(h+".npz")) as arr:
            for k in arr:
                field_arrays[k].append(arr[k])
    fields={k:np.concatenate(v,axis=0) for k,v in field_arrays.items()}
    ordering=np.argsort(fields["ix"])
    fields={k:v[ordering] for k,v in fields.items()}
    assert fields["ix"].tolist()==[i for i,_ in selected]
    digest=hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    np.savez_compressed(a.out/"geometry.npz",**fields,manifest_sha256=digest)
    aligned=(fields["rgb_mae"]<=1.)&(fields["rgb_bad_pixel_fraction"]<=.02)
    result=dict(total=len(selected),aligned=int(aligned.sum()),usable=int(fields["valid"].sum()),
                zero_rgb_difference=int((fields["rgb_mae"]==0).sum()),median_rgb_mae=float(np.median(fields["rgb_mae"])),
                maximum_rgb_mae=float(np.max(fields["rgb_mae"])),median_depth_m=float(np.nanmedian(fields["depth"])),
                manifest_sha256=digest)
    print(json.dumps(result),flush=True)
    (a.out/"geometry_audit.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("assets","manifest","out"):
        p.add_argument(key,type=Path)
    p.add_argument("--workers",type=int,default=3)
    p.add_argument("--house")
    main(p.parse_args())
