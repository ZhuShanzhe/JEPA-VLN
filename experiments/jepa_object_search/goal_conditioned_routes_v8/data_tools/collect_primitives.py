"""Observe an HSSD start and physically execute nine fixed alternatives for supervised route learning."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import habitat_sim as hs
from habitat_sim.utils.common import quat_from_coeffs
import numpy as np

OFFICIAL={"chair":"chair.n.01","bed":"bed.n.01","couch":"sofa.n.01","toilet":"toilet.n.02","potted_plant":"potted_plant.n.01","tv":"television_receiver.n.01"}
SAMPLES=(2,4,6,12,24)
HORIZONS=(4,12,24)
NAMES=("move_forward","turn_left","turn_right")


def plans():
    library={"straight":[0]*24}
    for name,turn,back in (("left",1,2),("right",2,1)):
        for n in (1,2):
            library[name+str(n*30)]=([turn]*n+[0]*24)[:24]
            library[name+str(n*30)+"_three_forward_back"]=([turn]*n+[0]*3+[back]*n+[0]*24)[:24]
    return library


def save_npz(path,**arrays):
    tmp=path.with_suffix(".pending.npz")
    np.savez_compressed(tmp,**arrays)
    os.replace(tmp,path)


def scene_targets(sim,source,extension,house,queries):
    with gzip.open(source/(house+".json.gz"),"rt") as f:
        data=json.load(f)["goals_by_category"]
    ext=json.loads((extension/(house+".goals.json")).read_bytes())
    by_name={}; dropped=set()
    rigids=sim.get_rigid_object_manager().get_objects_by_handle_substring("")
    for group in data.values():
        for g in group:
            if g["object_category"] not in OFFICIAL: continue
            if g["object_name"] not in rigids:
                dropped.add(g["object_category"]); continue
            by_name[g["object_name"]]=(g,OFFICIAL[g["object_category"]])
    by_name={k:(g,q) for k,(g,q) in by_name.items() if g["object_category"] not in dropped}
    aux_names=json.loads((extension/"query_map.json").read_bytes())
    for g in ext:
        key=g["object_name"]
        if key not in rigids or key in by_name:
            raise ValueError("Unmatched or duplicated derived instance")
        by_name[key]=(g,aux_names[g["object_category"]])
    # Semantic instance IDs are assigned afresh and map to class indices only inside the teacher.
    lookup=np.full(len(rigids)+1,len(queries),int)
    goals=[];classes=[]
    for sid,(name,body) in enumerate(sorted(rigids.items()),1):
        body.semantic_id=sid
        if name not in by_name:continue
        raw,word=by_name[name]
        cls=queries.index(word)
        goals.append(dict(object_id=sid,view_points=raw["view_points"]))
        classes.append(cls);lookup[sid]=cls
    return goals,np.asarray(classes),lookup,sorted(dropped)


def grid_labels(obs,lookup,n,cell):
    ids=obs["semantic"]
    cls=lookup[np.where((ids>=0)&(ids<len(lookup)),ids,0)]
    good=cls<n
    index=cls*49+cell
    depth=obs["depth"]
    def counts(mask):
        return np.bincount(index[mask],minlength=n*49).reshape(n,7,7).astype(np.uint16)
    return counts(good),counts(good & (depth>0)&(depth<=2.0))


def collect(a):
    sys.path.insert(0,str(a.renderer))
    from render_labels import simulation
    from opportunity_labels import ReportViewpoints
    os.environ.setdefault("HABITAT_SIM_LOG","quiet")
    extension=a.extended
    words=list(OFFICIAL.values())+sorted(json.loads((extension/"query_map.json").read_bytes()).values())
    assert len(set(words))==len(words)==16
    geom=simulation(a.house)
    branch=plans(); actions=np.array(list(branch.values()),dtype=np.int64)
    take=sorted(a.starts.glob("*/start*/manifest.json"))[:a.count]
    assert len(take)==a.count
    for d in ("raw","observed","labels","rows"):
        (a.dest/d).mkdir(exist_ok=True,parents=True)
    try:
        goals,cls,lookup,dropped=scene_targets(geom,a.official,extension,a.house,words)
        teacher=ReportViewpoints(geom.pathfinder,goals)
        n=len(words)
        bins=np.arange(256)*7//256
        cell=bins[:,None]*7+bins[None,:]
        cache={}
        def eligible(pos):
            key=tuple(np.asarray(pos,dtype=np.float32))
            if key not in cache:cache[key]=teacher.eligible(pos)
            return cache[key]
        for manifest in take:
            sid="primitives/"+a.split+"/"+a.house+"/"+manifest.parent.parent.name+"/"+manifest.parent.name
            uid=hashlib.sha256(sid.encode()).hexdigest()
            record=a.dest/"rows"/(uid+".json")
            if record.is_file():continue
            pose=np.asarray(json.loads(manifest.read_bytes())["frozen_pose_xyz_xyzw"])
            state=hs.AgentState()
            state.position=pose[:3];state.rotation=quat_from_coeffs(pose[3:])
            agent=geom.get_agent(0);agent.set_state(state)
            ctx=[];match=[];nearctx=[]
            for _ in range(12):
                obs=geom.get_sensor_observations()
                ctx.append(obs["rgb"][:,:,:3].copy())
                g,c=grid_labels(obs,lookup,n,cell);match.append(g);nearctx.append(c)
                agent.act("turn_left")
            # All views and semantics are rendered together, including at held-out starts.
            # External collector RGB is an additional alignment check, not a model input.
            with np.load(manifest.parent/"context.npz") as z:
                agreement=float(np.abs(ctx[0].astype(float)-z["rgb"][0]).mean())
            if agreement>4:raise ValueError((sid,"unexpected reference RGB deviation",agreement))
            start_hit=eligible(pose[:3]);rgb=[];far=[];close=[];gains=[];collisions=[];progress=[]
            for plan in actions:
                agent.set_state(state)
                hist=[];r=[];d=[];nr=[];col=[];movement=[]
                prev=np.asarray(agent.get_state().position);travel=0.
                for step,act in enumerate(plan,1):
                    blocked=agent.act(NAMES[act]); point=np.asarray(agent.get_state().position)
                    travel+=float(np.linalg.norm(point-prev));prev=point.copy()
                    movement.append(travel);col.append(bool(blocked))
                    hist.append(eligible(point))
                    if step in SAMPLES:
                        obs=geom.get_sensor_observations()
                        r.append(obs["rgb"][:,:,:3].copy())
                        g,c=grid_labels(obs,lookup,n,cell);d.append(g);nr.append(c)
                hist=np.asarray(hist)
                counts=[np.bincount(cls[hist[:h].any(0)&~start_hit],minlength=n) for h in HORIZONS]
                gains.append(counts);rgb.append(r);far.append(d);close.append(nr);progress.append(movement);collisions.append(col)
            raw=a.dest/"raw"/(uid+".npz");obsfile=a.dest/"observed"/(uid+".npz");lab=a.dest/"labels"/(uid+".npz")
            save_npz(obsfile,context_rgb=ctx,actions=actions,yaw=np.arange(12)*np.pi/6)
            # Kept separate so blind evaluation can hash/open only the observations.
            save_npz(raw,context_rgb=ctx,future_rgb=rgb,actions=actions,samples=SAMPLES)
            save_npz(lab,queries=words,query_present=np.bincount(cls,minlength=n)>0,
                     ctx_grid=match,ctx_near_grid=nearctx,future_grid=far,future_near_grid=close,gain=gains,
                     actual_progress=progress,collisions=collisions,samples=SAMPLES,horizons=HORIZONS,
                     baseline_reported_count=0,start_eligible_by_class=np.bincount(cls[start_hit],minlength=n))
            row=dict(sid=sid,house=a.house,split=a.split,raw=str(raw),observed=str(obsfile),labels=str(lab),
                     features=str(a.dest/"features"/(uid+".npz")),observed_features=str(a.dest/"observed_features"/(uid+".npz")),
                     catalog_exclusions=dropped,baseline_rgb_mae=agreement)
            tmp=record.with_suffix(".pending");tmp.write_text(json.dumps(row));os.replace(tmp,record)
            print("done",a.house,manifest.parent.name,flush=True)
    finally:geom.close()


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ["renderer","official","extended","starts","dest"]:ap.add_argument("--"+k,type=Path,required=True)
    ap.add_argument("--house",required=True);ap.add_argument("--split",required=True)
    ap.add_argument("--count",type=int,required=True)
    collect(ap.parse_args())
