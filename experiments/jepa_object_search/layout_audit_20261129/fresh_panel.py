"""Independent episode freeze and new-suite physical replay using actual RGB-D observations."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from panel import HERE, ROOT, ORIGINAL, SOURCE, read

DIR = HERE/"private_fresh"
ASSETS = Path("/root/autodl-tmp/jepa_vln_data_20261014/raw/hssd-hab")


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def freeze():
    import yaml
    raw = yaml.safe_load((ASSETS/"scene_splits.yaml").read_text())
    reference = read(SOURCE/"view_design.json")["cohorts"]
    excluded = {h.split("_")[0] for hs in reference.values() for h in hs}
    repaired = read(ASSETS.parent.parent/"manifests/material_compatibility.json")["referencing_scenes"]
    known = read(SOURCE/"house_history_audit.json")["by_scene"]
    options = [str(h) for h in raw["val"] if str(h).split("_")[0] not in excluded
               and str(h) not in repaired and (ASSETS/"scenes"/(str(h)+".scene_instance.json")).exists()]
    # Published IDs stay fixed following a metadata correction. The original lookup of
    # historical usage queried family stems in an index keyed by complete scene names.
    chosen = ["107734176_176000019","107734158_175999998","106878915_174887025"]
    assert set(chosen) <= set(options)
    assert len(chosen)==len({h.split("_")[0] for h in chosen})==3
    cfg = dict(houses=chosen, previous_history=[known[h] for h in chosen],
               split="Three frozen installed official validation houses; exclude current train/calibration/94 house family IDs and material overlays. These houses were used in older project experiments, and all local requests were resampled. Full-scene historical labels have been corrected; selected house IDs have not changed.",
               task="New derived local inspection episodes. First 24 initial cues per house with 38-368 pixels on the source image and <369 after a 30 degree scan, ordered by a fixed random seed. Local caps cycle 22/26/30; no future viewpoint or reachability test is used in selection.")
    p = HERE/"fresh_design.json"
    if p.exists():
        assert read(p)["houses"]==chosen
    p.write_text(json.dumps(cfg,indent=2)+"\n")


def prepare(house):
    from sample_views import sources, inventory, simulator, camera, ASSETS, ACTION, SOURCE_H
    sim = simulator(house)
    seed = int(digest("new-layout-raw-source-v1:"+house)[:8],16)%2**31
    sim.seed(seed)
    sim.pathfinder.seed(seed)
    directory = DIR/"starts"/house
    directory.mkdir(parents=True,exist_ok=True)
    try:
        ag = sim.get_agent(0)
        rows = sources(sim,inventory(sim,ASSETS),np.random.default_rng(seed),92)
        count = 0
        for category,oid,pos,yaw,cue in rows:
            obs = camera(sim,pos,yaw,SOURCE_H,0.)
            if not 38 <= int((obs["ids"]==oid).sum()) < 369:
                continue
            ag.act(ACTION[2])
            if int((sim.get_sensor_observations()["ids"]==oid).sum())>=369:
                continue
            np.savez(directory/(str(count)+".npz"), oid=oid,position=pos,yaw=yaw,box=cue[0],category=category,budget=(22,26,30)[count%3])
            count += 1
            if count==24:
                break
        assert count==24, (house,count)
    finally:
        sim.close()


def sim_worker(house,route,packet):
    from sample_views import simulator, inventory, camera, ASSETS, ACTION, SOURCE_H
    sim = simulator(house)
    try:
        inventory(sim,ASSETS)
        ag = sim.get_agent(0)
        source = None
        labels = []
        def store():
            if source is not None:
                where = DIR/route/"scoring"/house
                where.mkdir(parents=True,exist_ok=True)
                np.savez(where/(source+".npz"),packed=labels)
        for msg in sys.stdin:
            cmd, item = msg.split()
            if cmd=="source":
                store()
                source = item
                labels = []
                with np.load(DIR/"starts"/house/(source+".npz")) as s:
                    pos,yaw,oid = s["position"],float(s["yaw"]),int(s["oid"])
                obs = camera(sim,pos,yaw,SOURCE_H,0.)
                bump = False
            else:
                assert cmd=="act"
                act = int(item)
                prev = ag.state.position.copy()
                contact = ag.act(ACTION[act])
                bump = bool(act==1 and (contact or np.linalg.norm((ag.state.position-prev)[[0,2]])<.23))
                obs = sim.get_sensor_observations()
            delta = ag.state.position-pos
            q = ag.state.rotation
            a = 2*math.atan2(q.y,q.w)-yaw
            odom = [math.cos(yaw)*delta[0]-math.sin(yaw)*delta[2],-math.sin(yaw)*delta[0]-math.cos(yaw)*delta[2],math.atan2(math.sin(a),math.cos(a))]
            labels.append(np.packbits((obs["ids"]==oid).reshape(-1)))
            np.savez(packet,rgb=obs["rgb"][...,:3], depth=obs["depth"].squeeze(), pose=odom,bump=bump)
            print("observation_ready",flush=True)
        store()
    finally:
        sim.close()


class Continue:
    def __init__(self,*args):
        pass
    def observe(self,*args):
        return {"accept":False}


def drive(house,route):
    import torch
    from threadpoolctl import threadpool_limits
    from refined_inspection import load_inspection
    from verify_inspection_session import receive
    torch.set_num_threads(2)
    actor = load_inspection()
    actor.verifier = Continue
    if route == "grid":
        sys.path.insert(0,str(ROOT/"layout_detour_v2_20261129"))
        from grid_policy import GridPlanner
        actor.reroute = GridPlanner(actor.reroute)
    directory = DIR/route/"sensors"/house
    directory.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ,HABITAT_SIM_LOG="quiet",MAGNUM_LOG="quiet",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    env.pop("LD_LIBRARY_PATH",None)
    env.pop("PYTHONPATH",None)
    with tempfile.TemporaryDirectory(dir=directory) as tmp, threadpool_limits(limits=2):
        packet = Path(tmp)/"frame.npz"
        proc = subprocess.Popen([str(ROOT/"simulation/habitat-env/bin/python"),"-u",__file__,"sim","--house-id",house,"--route",route,"--packet",str(packet)],
                                stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env)
        try:
            for idx in range(24):
                with np.load(DIR/"starts"/house/(str(idx)+".npz")) as f:
                    box,budget = f["box"].copy(),int(f["budget"])
                images,dep,ods,motion,bumps = [],[],[],[],[]
                def observe(cmd,action):
                    frame = receive(proc,packet,cmd)
                    images.append(frame["rgb"]);dep.append(frame["depth"])
                    ods.append([frame[k] for k in ("x","z","yaw")])
                    motion.append(action);bumps.append(frame["bump"])
                    return frame
                src = observe("source "+str(idx),0)
                result = actor.run(src,box,budget-18,budget,lambda a:observe("act "+str(a),a),tmp)
                assert result["actions_used"]==len(motion)-1<=budget
                np.savez(directory/(str(idx)+".npz"),rgb=images,depth=dep,odom=ods,action=motion,bump=bumps,cue=box,budget=budget)
        finally:
            proc.stdin.close()
            if proc.wait(timeout=60)!=0:
                raise RuntimeError("fresh simulation error")


def verify(house,route):
    from memory_refit import RefinedMemory
    from strict_rescue import StrictRescueMemory
    from score_paired import box_grade
    from threadpoolctl import threadpool_limits
    directory = DIR/route
    out = []
    with threadpool_limits(limits=1):
        for idx in range(24):
            name = str(idx)+".npz"
            with np.load(directory/"sensors"/house/name) as f:
                rgb,depth,odom,box,budget = f["rgb"],f["depth"],f["odom"],f["cue"],int(f["budget"])
                bumps=int(f["bump"].sum())
            submits = {}
            for key, cls in (("old",RefinedMemory),("strict",StrictRescueMemory)):
                verifier=cls(rgb[0],depth[0],box)
                for t,(im,z,p) in enumerate(zip(rgb,depth,odom)):
                    e=verifier.observe(im,z,p)
                    if e["accept"]:
                        submits[key]=(t,[int(x) for x in e["box"]])
                        break
            with np.load(directory/"scoring"/house/name) as f:
                masks=np.unpackbits(f["packed"],axis=1).reshape(-1,192,192)
            assert len(rgb)==len(masks)<=budget+1
            row={"ever_clear":bool((masks.sum((1,2))>=369).any()),"full_steps":len(rgb)-1,"bumps":bumps}
            for k in ("old","strict"):
                if k not in submits:
                    status,step="unreported",len(rgb)-1
                else:
                    step,bb=submits[k]
                    n,v,iou=box_grade(masks[step],bb)
                    status="correct" if n>=369 and 2*v>=n and iou>=.30 else "false"
                row[k]={"outcome":status,"cost":step}
            out.append(row)
    (directory/("score_"+house+".json")).write_text(json.dumps(out))


def summarize():
    totals={}
    rooms=[]
    moves=Counter()
    paired=Counter()
    for house in read(HERE/"fresh_design.json")["houses"]:
        per={}
        paths={}
        for route in ("baseline","grid"):
            paths[route]=read(DIR/route/("score_"+house+".json"))
            for k in ("old","strict"):
                name=route+"_"+k
                counts=Counter()
                for row in paths[route]:
                    counts[row[k]["outcome"]]+=1
                    counts["actions"]+=row[k]["cost"]
                    counts["ever_clear"]+=int(row["ever_clear"])
                    counts["full_path_bumps"]+=row["bumps"]
                per[name]=counts
                totals.setdefault(name,Counter()).update(counts)
        for prev,cur in zip(paths["baseline"],paths["grid"]):
            paired[prev["old"]["outcome"]+"->"+cur["strict"]["outcome"]]+=1
            moves[str(prev["ever_clear"])+"->"+str(cur["ever_clear"])]+=1
        rooms.append(per)
    assert all(sum(row[k] for k in ("correct","false","unreported"))==72 for row in totals.values())
    (HERE/"fresh_results.json").write_text(json.dumps(dict(groups=72,houses=3,results=totals,per_house=rooms,paired=paired,route_opportunities=moves),indent=2)+"\n")
    print(totals)


if __name__=="__main__":
    os.umask(0o077)
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("freeze","prepare","sim","drive","score","summarize"))
    p.add_argument("--house",type=int,choices=(0,1,2))
    p.add_argument("--house-id")
    p.add_argument("--route",choices=("grid","baseline"))
    p.add_argument("--packet")
    a=p.parse_args()
    if a.mode=="freeze":
        freeze()
    elif a.mode=="summarize":
        summarize()
    else:
        house=a.house_id or read(HERE/"fresh_design.json")["houses"][a.house]
        if a.mode=="prepare":prepare(house)
        elif a.mode=="sim":sim_worker(house,a.route,a.packet)
        elif a.mode=="drive":drive(house,a.route)
        else:verify(house,a.route)
