"""Collect household category search forks with independent panoramic invisibility checks."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
import habitat_sim as hs
from habitat_sim.utils.common import quat_from_angle_axis
import magnum as mn

ACT=("move_forward","turn_left","turn_right")
WORDS=("chair.n.01","bed.n.01","sofa.n.01","toilet.n.02","television_receiver.n.01","potted_plant.n.01","lamp.n.01","wall_art.n.01","clock.n.01","mirror.n.01","picture_frame.n.01","basket.n.01","cabinet.n.01","shelf.n.01","stool.n.01","table.n.02","vase.n.01","book.n.01","desk.n.01","refrigerator.n.01","bathtub.n.01","sink.n.01","oven.n.01","chandelier.n.01","ceiling_lamp.n.01","floor_lamp.n.01","armchair.n.01","plant.n.02","nightstand.n.01","bookcase.n.01")
def simulate(asset,house):
    config=hs.SimulatorConfiguration()
    config.scene_dataset_config_file=str(asset/"hssd-hab.scene_dataset_config.json")
    config.scene_id=house;config.enable_physics=True;config.allow_sliding=False
    ag=hs.agent.AgentConfiguration(height=1.4,radius=.18)
    for name,typ in (("rgb",hs.SensorType.COLOR),("depth",hs.SensorType.DEPTH),("ids",hs.SensorType.SEMANTIC)):
        s=hs.CameraSensorSpec();s.uuid=name;s.sensor_type=typ;s.resolution=[128,128]
        s.position=mn.Vector3(0,1.25,0);s.hfov=mn.Deg(90);ag.sensor_specifications.append(s)
    ag.action_space={a:hs.agent.ActionSpec(a,hs.agent.ActuationSpec(amount=v)) for a,v in zip(ACT,(.25,30.,30.))}
    sim=hs.Simulator(hs.Configuration(config,[ag]))
    nav=hs.NavMeshSettings();nav.set_defaults();nav.agent_height=1.4;nav.agent_radius=.18;nav.include_static_objects=True
    if not sim.recompute_navmesh(sim.pathfinder,nav):raise ValueError("navmesh")
    return sim

def semantics(sim,asset):
    with (asset/"metadata/fpmodels-with-decomposed.csv").open() as f:
        cat={r["id"]:(r["main_wnsynsetkey"] or r["wnsynsetkey"]) for r in csv.DictReader(f)}
    ids=[-1]
    for i,(name,obj) in enumerate(sorted(sim.get_rigid_object_manager().get_objects_by_handle_substring("").items()),1):
        matches=re.findall(r"[a-fA-F0-9]{40}",name)
        word=next((cat[x] for x in matches if x in cat),None)
        ids.append(WORDS.index(word) if word in WORDS else -1);obj.semantic_id=i
    if max(ids)<0:raise ValueError("unmapped object inventory")
    return np.array(ids)

def world_state(pos,yaw):
    s=hs.AgentState();s.position=np.array(pos);s.rotation=quat_from_angle_axis(yaw,np.array([0.,1.,0.]));return s

def nav_path(pf,a,b):
    path=hs.ShortestPath();path.requested_start=a;path.requested_end=b
    if not pf.find_path(path):return None
    return path

def fork(sim, start, rng):
    """Select three unlabeled endpoints by reachable distance and directional spread only."""
    pf=sim.pathfinder;pool=[]
    for i in range(96):
        b=pf.get_random_navigable_point_near(start,5.5)
        if not np.isfinite(b).all() or abs(b[1]-start[1])>.45:continue
        d=b-start;l=np.linalg.norm(d[[0,2]])
        if not 2.0<l<5.2:continue
        p=nav_path(pf,start,b)
        if p and 3.25<=p.geodesic_distance<=5.25:pool.append((b,d[[0,2]]/l))
        if len(pool)>=20:break
    if len(pool)<3:return None
    chosen=[pool[0]]
    while len(chosen)<3:
        k=np.argmin([max(np.dot(e[1],v[1]) for v in chosen) for e in pool])
        v=pool[k]
        if max(np.dot(v[1],e[1]) for e in chosen)>.86:return None
        chosen.append(v)
    rng.shuffle(chosen);return [c[0] for c in chosen]

def collect_one(sim,lookup,start,yaw,ends):
    agent=sim.get_agent(0);base=world_state(start,yaw)
    panor=[];seen_ids=set()
    for i in range(12):
        agent.set_state(world_state(start,yaw+i*math.pi/6))
        obs=sim.get_sensor_observations()
        seen_ids.update(np.unique(obs["ids"]).tolist());panor.append(obs["rgb"][:,:,:3].copy())
    seen={lookup[i] for i in seen_ids if 0<=i<len(lookup)};n=len(lookup)
    ctx=np.stack([panor[11],panor[0],panor[1]])
    plans=[];future=[];gains=[];collision=[];instance_frames=[]
    for k,end in enumerate(ends):
        agent.set_state(base)
        follower=sim.make_greedy_follower(agent_id=0,goal_radius=.22,stop_key="stop",forward_key=ACT[0],left_key=ACT[1],right_key=ACT[2])
        try:acts=list(follower.find_path(end))
        except hs.errors.GreedyFollowerError:return None
        acts=[ACT.index(a) for a in acts if a in ACT]
        if len(acts)<8:return None
        # Identical 24 action budget; scan by 30 degrees if reaching the waypoint early.
        acts=(acts+[1,1,2,2,2,2,1,1]*3)[:24]
        agent.set_state(base);frames=[];hit=set();cf=[];per_frame=[]
        for t,a in enumerate(acts,1):
            c=agent.act(ACT[a]);cf.append(bool(c) if a==0 else False)
            if t in (6,12,18,24):
                obs=sim.get_sensor_observations();frames.append(obs["rgb"][:,:,:3].copy())
                sem=obs["ids"];valid=(sem>0)&(sem<n)&np.isfinite(obs["depth"])&(obs["depth"]>0)&(obs["depth"]<=2.)
                count=np.bincount(sem[valid].astype(int), minlength=n)
                visible=np.flatnonzero(count>=32)
                hit.update(visible.tolist());per_frame.append(visible.tolist())
        plans.append(acts);future.append(frames);collision.append(cf);instance_frames.append(per_frame)
        gains.append(np.bincount([lookup[i] for i in hit if lookup[i]>=0],minlength=len(WORDS)))
    y=np.stack(gains);valid=[q for q in range(len(WORDS)) if q not in seen and y[:,q].max()>0 and y[:,q].min()==0]
    if not valid:return None
    return dict(ctx=ctx,headings=np.array([-30,0,30]),actions=np.array(plans,dtype=np.uint8),
                future=np.asarray(future),collided=np.asarray(collision),words=np.array(WORDS)[valid],y=y[:,valid],
                house_instances=np.array([(lookup==q).sum() for q in valid]),start_pixels=np.zeros(len(valid),int),
                source="hssd",group="no_category_in_start_panorama",future_steps=np.array([6,12,18,24]))

def run(asset,root,split,house,target,max_trials):
    dest=root/split/house;dest.mkdir(parents=True,exist_ok=True)
    out=dest/"receipt.json"
    if out.exists() and json.loads(out.read_text())["maximum_trials"]==max_trials:return
    seed=int(hashlib.sha256(("v11natural|"+house).encode()).hexdigest()[:8],16)%2147483647
    rng=np.random.default_rng(seed); sim=simulate(asset,house);sim.pathfinder.seed(seed)
    poses=[];saved=0;eligible=0;tested=0;tested_starts=[]
    try:
        lookup=semantics(sim,asset)
        for trial in range(max_trials):
            start=sim.pathfinder.get_random_navigable_point()
            if not np.isfinite(start).all() or any(np.linalg.norm(start-np.asarray(p))<.9 for p in poses):continue
            yaw=rng.uniform(-math.pi,math.pi);ends=fork(sim,start,rng)
            if ends is None:continue
            tested+=1;record=collect_one(sim,lookup,start,yaw,ends)
            if record is None:continue
            poses.append(start.tolist());origin=f"{house}-{trial:04d}"
            file=dest/(origin+".npz");temp=file.with_suffix(".partial.npz")
            if file.exists():
                with np.load(file) as old:
                    if any(not np.array_equal(old[k],v) for k,v in record.items()):
                        raise ValueError("physical prefix disagrees with previous slice")
            else:
                np.savez_compressed(temp,**record,house=house,origin=origin);temp.replace(file)
            tested_starts.append(dict(origin=origin,position=start.tolist(),yaw=yaw,ends=[x.tolist() for x in ends]))
            eligible+=len(record["words"]);saved+=1
            if saved>=target:break
    finally:sim.close()
    (dest/"private_replay.json").write_text(json.dumps(tested_starts))
    out.write_text(json.dumps(dict(house=house,split=split,maximum_trials=max_trials,attempted=trial+1,valid_forks_tested=tested,origins=saved,queries=eligible)))
    print(out.read_text(),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("asset","root"):p.add_argument(k,type=Path)
    p.add_argument("split");p.add_argument("house");p.add_argument("--target",type=int,default=12);p.add_argument("--trials",type=int,default=75)
    a=p.parse_args();run(a.asset,a.root,a.split,a.house,a.target,a.trials)
