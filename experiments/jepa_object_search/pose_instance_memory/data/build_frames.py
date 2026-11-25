"""Frozen HSSD co-visible, singleton and turned-away short camera sequences."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

CLOCKS = {"clock","wall_clock","alarm_clock","mantel_clock","grandfather_clock"}
PRIORITY = ["clock","wall_art","picture_frame","chair","mirror","lamp","vase","stool"]
SIDE = 384
FOCAL = SIDE/2/math.tan(math.radians(75/2))
K = [[FOCAL,0,SIDE/2],[0,FOCAL,SIDE/2],[0,0,1]]
MIN_PIXELS, MIN_SIDE, MIN_FILL = 170, 12, .10


def stable(*parts):
    return hashlib.sha256(("v21-frame-v1|"+"|".join(map(str,parts))).encode()).hexdigest()


def read_rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def open_sim(assets,house):
    import habitat_sim as hs
    import magnum as mn
    cfg=hs.SimulatorConfiguration()
    cfg.scene_dataset_config_file=str(assets/"hssd-hab.scene_dataset_config.json")
    cfg.scene_id=house
    cfg.enable_physics=True
    agent=hs.agent.AgentConfiguration(height=1.4,radius=.18)
    for key,typ in (("rgb",hs.SensorType.COLOR),("depth",hs.SensorType.DEPTH),("ids",hs.SensorType.SEMANTIC)):
        spec=hs.CameraSensorSpec()
        spec.uuid,spec.sensor_type,spec.resolution=key,typ,[SIDE,SIDE]
        spec.position,spec.hfov=mn.Vector3(0,1.25,0),mn.Deg(75)
        agent.sensor_specifications.append(spec)
    sim=hs.Simulator(hs.Configuration(cfg,[agent]))
    nav=hs.NavMeshSettings()
    nav.set_defaults()
    nav.agent_height,nav.agent_radius,nav.include_static_objects=1.4,.18,True
    if not sim.recompute_navmesh(sim.pathfinder,nav):
        raise RuntimeError("navmesh unavailable")
    return sim


def annotations(semantic,inventory):
    boxes=[]
    for sid in np.unique(semantic):
        if sid not in inventory:
            continue
        yy,xx=np.where(semantic==sid)
        if len(xx)<MIN_PIXELS:
            continue
        xyxy=list(map(int,(xx.min(),yy.min(),xx.max()+1,yy.max()+1)))
        width,height=xyxy[2]-xyxy[0],xyxy[3]-xyxy[1]
        if min(width,height)<MIN_SIDE or len(xx)/(width*height)<MIN_FILL:
            continue
        boxes.append(dict(**inventory[sid],xyxy=xyxy,pixels=len(xx)))
    return boxes


def choose(candidates):
    selected=[]
    used=set()
    for group,limit in (("multiple",3),("single",2),("turned",1)):
        pool=[d for d in candidates if group=="turned" or (len(d["boxes"])>=2)==(group=="multiple")]
        classes=set()
        for k in range(limit):
            remaining=[d for d in pool if d["key"] not in used]
            if not remaining:
                break
            def rank(d):
                cat=d["row"]["category"]
                priority=PRIORITY.index(cat) if cat in PRIORITY else len(PRIORITY)
                return (cat in classes,priority if group!="turned" else 0,
                        not d["same_asset_close"] if group=="multiple" else False,
                        d["distance"]>1.25 if group=="multiple" else False,
                        stable(d["key"],group))
            d=min(remaining,key=rank)
            selected.append((group,d))
            classes.add(d["row"]["category"])
            used.add(d["key"])
    return selected


def one(a):
    sys.path.insert(0,str(a.previous/"model"))
    sys.path.insert(0,str(a.previous/"collect"))
    import collect_views
    from replay_depth import observe,rgb_error
    from habitat_sim.utils.common import quat_rotate_vector
    house=a.house
    allrows=[]
    for name in ("data","clocks"):
        manifest=a.previous/name/"manifest.jsonl"
        if manifest.exists():
            allrows.extend((r,name) for r in read_rows(manifest) if r["house"]==house)
    if not allrows:
        raise ValueError(house)
    split=allrows[0][0]["split"]
    sim=open_sim(a.assets,house)
    try:
        collect_views.WORDS.update(CLOCKS)
        groups=collect_views.inventory(sim,a.assets)
        cat_index=defaultdict(dict)
        for subtype,objs in groups.items():
            cat="clock" if subtype in CLOCKS else subtype
            for sid,asset,center in objs:
                cat_index[cat][sid]=dict(instance=f"{house}:{sid}",asset=asset,subtype=subtype,
                                         world_center=center.tolist())
        candidates=[]
        for row,source in allrows:
            cat=row["category"]
            if cat not in cat_index:
                continue
            obs,state=observe(sim,row)
            boxes=annotations(obs["ids"],cat_index[cat])
            target=[b for b in boxes if b["instance"]==row["instance"]]
            if len(target)!=1 or target[0]["xyxy"]!=row["bbox"]:
                raise RuntimeError(f"V20 semantic registration mismatch: {house}/{row['instance']}")
            close=False
            distance=float("inf")
            for first,second in combinations(boxes,2):
                gap=math.dist(first["world_center"],second["world_center"])
                close|=gap<=1.25 and first["asset"]==second["asset"]
                distance=min(distance,gap)
            key=stable(row["camera_xyz"],row["camera_heading_deg"],row["pitch_deg"])
            candidates.append(dict(row=row,source=source,boxes=boxes,distance=distance,same_asset_close=close,key=key))
        frames=[]
        for seqidx,(group,d) in enumerate(choose(candidates)):
            src=d["row"]
            eye=np.array(src["camera_xyz"])
            old=np.asarray(eye-[0,1.25,0],dtype=np.float32)
            heading=src["camera_heading_deg"]
            sideways=np.array([-math.sin(math.radians(heading)),0.,math.cos(math.radians(heading))])
            sid=f"{house}_{seqidx:02}"
            for view,(offset,shift) in enumerate(((0.,0.),(-.24,-11.),(.24,11.))):
                nav=np.asarray(sim.pathfinder.try_step(old,(old+offset*sideways).astype(np.float32))) if offset else old
                if not np.isfinite(nav).all() or np.linalg.norm(nav-old)>.38:
                    nav=old.copy()
                row=dict(src,camera_xyz=(nav+[0,1.25,0]).tolist(),
                         camera_heading_deg=heading+shift+(180 if group=="turned" else 0))
                if view==0:
                    row["camera_xyz"]=src["camera_xyz"]
                obs,state=observe(sim,row)
                if view==0 and group!="turned":
                    if rgb_error(obs["rgb"],src,a.previous/d["source"])[0]!=0:
                        raise RuntimeError("source RGB differs")
                pose=np.eye(4)
                rotation=np.column_stack([quat_rotate_vector(state.rotation,v) for v in np.eye(3)])
                pose[:3,:3]=rotation@np.diag([1.,-1.,-1.])
                pose[:3,3]=state.position
                boxes=annotations(obs["ids"],cat_index[row["category"]])
                frameid=f"{sid}_{view}"
                relative=Path("images")/split/house
                path=a.out/relative
                path.mkdir(parents=True,exist_ok=True)
                rgb=relative/(frameid+".jpg")
                depth=relative/(frameid+".npy")
                Image.fromarray(obs["rgb"][:,:,:3]).save(a.out/rgb,quality=93)
                np.save(a.out/depth,obs["depth"].astype(np.float32))
                frames.append(dict(frame_id=frameid,split=split,house=house,sequence=sid,view=view,
                                   stratum=group,category=row["category"],rgb=rgb.as_posix(),depth=depth.as_posix(),
                                   K=K,T_world_from_camera=pose.tolist(),camera_heading_deg=row["camera_heading_deg"],
                                   boxes=boxes,source_rgb_checked=(view==0 and group!="turned")))
    finally:
        sim.close()
    out=a.out/"shards"/(house+".jsonl")
    tmp=out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r,separators=(",",":"),allow_nan=False)+"\n" for r in frames))
    tmp.replace(out)
    print(house,len(frames),flush=True)


def main(a):
    if a.house:
        return one(a)
    bysplit=defaultdict(set)
    for r in read_rows(a.previous/"data/manifest.jsonl"):
        bysplit[r["split"]].add(r["house"])
    splits={k:sorted(v,key=lambda s:stable(s,"cohort"))[:10 if k=="train" else len(v)] for k,v in bysplit.items()}
    assert list(map(len,[splits["train"],splits["val"],splits["test"]]))==[10,6,16]
    assert sum(len(s) for s in splits.values())==len(set.union(*(set(v) for v in splits.values())))
    for d in ("logs","shards"):
        (a.out/d).mkdir(parents=True,exist_ok=True)
    schedule=[s[i] for i in range(max(map(len,splits.values()))) for s in [splits["test"],splits["val"],splits["train"]] if i<len(s)]
    protocol=dict(splits=splits,source=stable(Path(__file__).read_text()),cohorts=["multiple:3","single:2","turned:1"])
    dest=a.out/"frozen_protocol.json"
    if dest.exists() and json.loads(dest.read_text())!=protocol:
        raise RuntimeError("frozen configuration differs")
    dest.write_text(json.dumps(protocol))
    def worker(h):
        if (a.out/"shards"/(h+".jsonl")).exists():
            return
        with (a.out/"logs"/(h+".log")).open("a") as f:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),str(a.assets),str(a.previous),str(a.out),"--house",h],
                           stdout=f,stderr=f,check=True)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(worker,schedule))
    allrows=[r for h in schedule for r in read_rows(a.out/"shards"/(h+".jsonl"))]
    assert len(set(r["frame_id"] for r in allrows))==len(allrows)
    path=a.out/"frames.jsonl"
    tmp=path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r,separators=(",",":"),allow_nan=False)+"\n" for r in allrows))
    tmp.replace(path)
    from audit_frames import audit
    report=audit(a.out)
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("assets","previous","out"):
        p.add_argument(key,type=Path)
    p.add_argument("--house")
    p.add_argument("--workers",type=int,default=4)
    main(p.parse_args())
