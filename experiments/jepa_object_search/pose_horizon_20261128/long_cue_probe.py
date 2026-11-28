"""Physically replay frozen long base actions with a source-visible, fixed target instance."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RAW=HERE/"private_long_cue_render"
OUT=HERE/"private_long_cue"
TAPS=(12,24,52,80,108)
VIEWS=((1.31,0.),(.85,20.),(1.65,-20.))
SOURCES=dict(test="fresh",val="calibration",train="train")


def replay(split,house):
    from pose_data import (simulator,inventory,ASSETS,S,camera,cue_inputs,SOURCE_H,features,write_rgb)
    folder=RAW/(split+"_"+house)
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/"samples.json"
    if path.exists():return
    with np.load(HERE/"private_long"/(split+"-"+house+".npz")) as f:
        keep=(f["node"]=="r")&((f["action"]!=0).sum(1)==108)
        origins=f["origin"][keep]
        actions=f["action"][keep]
    statefile=ROOT/"jepa_env_v13/simulator/trees"/SOURCES[split]/house/"private_states.json"
    starts=json.loads(statefile.read_text())[:4]
    sim=simulator(house)
    ag=sim.get_agent(0)
    rows=[]
    try:
        inventory_map={sid:category for category,items in inventory(sim,ASSETS).items()
                       for sid,_,_ in items}
        for start in starts:
            orig=start["origin"]
            p=np.asarray(start["position"],dtype=np.float32)
            yaw=start["yaw"]
            src=camera(sim,p,yaw,SOURCE_H,0.)
            source=src["rgb"].copy()
            ids=src["ids"].copy()
            dep=src["depth"].squeeze().copy()
            values,counts=np.unique(ids,return_counts=True)
            eligible=sorted(((int(count),int(i)) for i,count in zip(values,counts)
                             if i in inventory_map and 40<=count<=.31*S*S),key=lambda k:(-k[0],k[1]))
            cues=[]
            for _,sid in eligible:
                inputs=cue_inputs(ids==sid,dep)
                if inputs is None:continue
                box,anchor,box_info=inputs
                l,t,r,b=box
                file=f"{orig}_{sid}_cue.jpg"
                write_rgb(source[t:b,l:r],folder/file)
                cues.append((sid,anchor,box_info,file))
                if len(cues)==2:break
            if not cues:continue
            srcname=f"{orig}_src.jpg"
            write_rgb(source,folder/srcname)
            plans=actions[origins==orig]
            assert plans.shape==(3,108)
            for plan_id,plan in enumerate(plans):
                camera(sim,p,yaw,SOURCE_H,0.)
                collision=False
                for step,a in enumerate(plan,1):
                    before=ag.state.position.copy()
                    ag.act({1:"move_forward",2:"turn_left",3:"turn_right"}[a])
                    endpoint=ag.state.position.copy()
                    if a==1 and np.linalg.norm((endpoint-before)[[0,2]])<.23:
                        collision=True
                    if step not in TAPS:continue
                    prefix=plan[:step].tolist()
                    _,_,x,z,theta=features(prefix,SOURCE_H,0.,cues[0][1],cues[0][2],dep)
                    ideal=p+np.array([x*math.cos(yaw)-z*math.sin(yaw),0,-x*math.sin(yaw)-z*math.cos(yaw)])
                    err=float(np.linalg.norm((ideal-endpoint)[[0,2]]))
                    reach=not collision and err<.14
                    original=ag.get_state()
                    for view,(height,pitch) in enumerate(VIEWS):
                        obs=camera(sim,endpoint,yaw+theta,height,pitch) if reach else None
                        dest=f"{orig}_{plan_id}_{step}_{view}.jpg" if reach else ""
                        if dest:write_rgb(obs["rgb"],folder/dest)
                        for sid,anchor,box_info,crop in cues:
                            feat,safe,*_=features(prefix,height,pitch,anchor,box_info,dep)
                            pix=int((obs["ids"]==sid).sum()) if reach else -1
                            rows.append(dict(source=srcname,cue=crop,future=dest,
                                origin=orig,group=f"{orig}/{sid}/h{step}",category=inventory_map[sid],
                                plan_id=plan_id,pose=feat,action=prefix,horizon=step,safe_mask=safe,
                                reach=int(reach),had_collision=collision,end_error_m=err,at_position=err<.14,
                                height=height,pitch_deg=pitch,native_mask=view==0,pixels=pix,
                                visible=int(pix>=math.ceil(.01*S*S)) if reach else -1))
                    ag.set_state(original,reset_sensors=False,infer_sensor_states=False)
    finally:
        sim.close()
    tmp=path.with_suffix(".partial")
    tmp.write_text(json.dumps(rows,allow_nan=False))
    tmp.replace(path)
    print("collected",split,house,len(rows),flush=True)


def encode(split,house,encoder):
    from pose_data_contract import FEATURES
    path=OUT/(split+"_"+house+".npz")
    if path.exists():
        with np.load(path,allow_pickle=False) as previous:
            if previous["had_collision"].dtype==np.bool_:return
    directory=RAW/path.stem
    rows=json.loads((directory/"samples.json").read_text())
    files=sorted({r[k] for r in rows for k in ("source","cue","future") if r[k]})
    lookup={"":np.zeros(1024,np.float16)}
    for k in range(0,len(files),64):
        images=[]
        for name in files[k:k+64]:
            with Image.open(directory/name) as im:images.append(np.asarray(im.convert("RGB")))
        lookup.update(zip(files[k:k+64],encoder.encode(images)))
    types=dict(origin=str,group=str,category=str,plan_id=np.int16,horizon=np.int16,
               safe_mask=bool,reach=np.int8,visible=np.int8,at_position=bool,pixels=np.int32,
               height=np.float32,pitch_deg=np.float32,native_mask=bool,end_error_m=np.float32,
               had_collision=bool)
    array={k:np.asarray([r[k] for r in rows],dtype=d) for k,d in types.items()}
    n=len(rows)
    a=np.zeros((n,108),np.uint8)
    for i,r in enumerate(rows): a[i,:r["horizon"]]=r["action"]
    array.update(action=a,context=np.asarray([np.r_[lookup[r["source"]],lookup[r["cue"]]] for r in rows],
                                            dtype=np.float16).reshape(n,2048),
                 future=np.asarray([lookup[r["future"]] for r in rows],dtype=np.float16).reshape(n,1024),
                 future_valid=array["visible"]>=0,
                 pose=np.asarray([r["pose"] for r in rows],np.float32).reshape(n,20),
                 feature_names=np.asarray(FEATURES),house=np.repeat(house,n),split=np.repeat(split,n))
    temp=path.with_suffix(".partial.npz")
    np.savez_compressed(temp,**array)
    temp.replace(path)
    print("encoded",split,house,n,flush=True)


def main():
    os.umask(0o077)
    p=argparse.ArgumentParser()
    p.add_argument("--worker",type=int,default=0)
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--split",default="test,val,train")
    p.add_argument("--encode",action="store_true")
    args=p.parse_args()
    cohort=ROOT/"jepa_env_v13/simulator"
    c=json.loads((cohort/"cohort.json").read_text())
    houses=dict(train=json.loads((cohort/"training_cohort.json").read_text())["train"],
                val=c["calibration"],test=c["fresh"])
    pending=[(s,h) for s in args.split.split(",") for h in houses[s]][args.worker::args.workers]
    if args.encode:
        import time
        import torch
        from long_data import FrozenImageEncoder
        torch.set_num_threads(2)
        encoder=FrozenImageEncoder()
        OUT.mkdir(exist_ok=True)
        last=time.monotonic()
        while pending:
            completed=[]
            for s,h in pending:
                if (RAW/(s+"_"+h)/"samples.json").exists():
                    encode(s,h,encoder);completed.append((s,h))
            if completed:last=time.monotonic()
            pending=[row for row in pending if row not in completed]
            if not completed:
                if time.monotonic()-last>600:raise RuntimeError(f"pending raw houses: {pending}")
                time.sleep(2)
    else:
        for s,h in pending:replay(s,h)

if __name__=="__main__":main()
