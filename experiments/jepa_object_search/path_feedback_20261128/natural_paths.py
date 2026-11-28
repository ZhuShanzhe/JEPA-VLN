"""Prospective house manifest, winding waypoint trajectories and fixed outcome-independent perturbations."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"pose_horizon_20261128"))
from pose_data import ASSETS,houses,simulator,camera,features,SOURCE_H
from path_geometry import from_depth,nominal,path_features,NAMES

ACT={1:"move_forward",2:"turn_left",3:"turn_right"}
TAPS=(12,24,52,80,108)
MANIFEST=HERE/"natural_design.json"


def freeze():
    groups=houses()
    exclude={h.split("_")[0] for hh in groups.values() for h in hh}
    original=json.loads((HERE.parent/"local_target_20261128/eval_family/panel.json").read_text())
    exclude.update(e["scene"].split("_")[0] for e in original)
    pool=[]
    for line in (ASSETS/"scene_splits.yaml").read_text().splitlines():
        if line.endswith(":"):part=line[:-1].strip()
        elif line.startswith("- ") and part=="train":
            h=line[2:].strip().strip("'\"")
            if h.split("_")[0] not in exclude and (ASSETS/"scenes"/(h+".scene_instance.json")).exists():
                pool.append(h)
    blind=[]
    for h in sorted(pool,key=lambda h:hashlib.sha256(("pf_blind_v1:"+h).encode()).hexdigest()):
        family=h.split("_")[0]
        if family in exclude:continue
        blind.append(h); exclude.add(family)
        if len(blind)==6:break
    assert len(blind)==6
    record=dict(cohorts={**groups,"blind":blind},starts=18,horizons=TAPS,
                design="Random navigable start and yaw, three or more same-floor random local waypoints, nominal route comes from offline Habitat greedy follower. No contact/outcome selection. Evaluate unmodified physical actions and three deterministic same-length turn/advance perturbations.",
                blind_selection="First six unique building families ranked by SHA256(pf_blind_v1: + ID) in installed official HSSD train excluding original 30 and previous eight. Frozen prior to collecting this panel.")
    output=json.dumps(record,indent=2)+"\n"
    if MANIFEST.exists():assert MANIFEST.read_text()==output
    else:MANIFEST.write_text(output)


def natural(sim,ag,start,yaw,rng):
    camera(sim,start,yaw,SOURCE_H,0)
    actions=[]
    # These privileged waypoints generate proposals offline. No waypoints, poses or navmesh enter the actor.
    for _ in range(100):
        p=np.array(ag.state.position)
        dest=np.asarray(sim.pathfinder.get_random_navigable_point_near(p,6.0)).copy()
        if not np.isfinite(dest).all() or abs(dest[1]-p[1])>.40 or np.linalg.norm((dest-p)[[0,2]])<1.5:continue
        follower=sim.make_greedy_follower(agent_id=0,goal_radius=.22,stop_key="stop",
                        forward_key=ACT[1],left_key=ACT[2],right_key=ACT[3])
        import habitat_sim
        try:leg=[a for a in follower.find_path(dest) if a in ACT.values()]
        except habitat_sim.errors.GreedyFollowerError:continue
        for a in leg:
            actions.append({"move_forward":1,"turn_left":2,"turn_right":3}[a])
            ag.act(a)
            if len(actions)==108:return actions
    return None


def variants(base,rng):
    n=len(base)
    turns=[i for i in range(max(1,n//8),max(2,3*n//4)) if base[i]!=1]
    ix=int(rng.choice(turns)) if turns else n//3
    sign=base[ix] if base[ix]!=1 else int(rng.choice([2,3]))
    extra=int(rng.integers(2,6))
    yield "none",base
    yield "forward_extra",(base[:ix]+[1]*extra+base[ix:])[:n]
    flip=base.copy();flip[ix]=5-sign
    yield "flip_turn",flip
    # Temporary sideward deflection; yaw restored before resuming the nominal suffix.
    yield "side_shift",(base[:ix]+[sign]*2+[1]*3+[5-sign]*2+base[ix:])[:n]


def run(split,h,ns):
    folder=HERE/"private_natural";folder.mkdir(exist_ok=True)
    path=folder/(split+"_"+h+".npz")
    if path.exists():return
    seed=int(hashlib.sha256(("pf_natural_v1:"+h).encode()).hexdigest()[:8],16)%(2**31-1)
    rng=np.random.default_rng(seed);sim=simulator(h)
    sim.seed(seed);sim.pathfinder.seed(seed)
    ag=sim.get_agent(0);rows=[];used=[]
    try:
        for trial in range(ns*12):
            p=np.asarray(sim.pathfinder.get_random_navigable_point()).copy()
            if not np.isfinite(p).all() or any(np.linalg.norm(q-p)<.65 for q in used):continue
            yaw=float(rng.uniform(-math.pi,math.pi))
            acts=natural(sim,ag,p,yaw,rng)
            if acts is None:continue
            idx=len(used);used.append(p)
            d=camera(sim,p,yaw,SOURCE_H,0)["depth"].squeeze().copy()
            state=ag.get_state();g=from_depth(d)
            for tap in TAPS:
                for kind,plan in variants(acts[:tap],rng):
                    ag.set_state(state,reset_sensors=False,infer_sensor_states=False)
                    blocked=0
                    for k,a in enumerate(plan,1):
                        prev=ag.state.position.copy()
                        contact=ag.act(ACT[a])
                        if a==1 and (contact or np.linalg.norm((ag.state.position-prev)[[0,2]])<.23):
                            blocked=k
                            break
                    ph=path_features(plan,g)
                    x,z=ph[-1,:2]
                    target=p+np.array([x*math.cos(yaw)-z*math.sin(yaw),0,-x*math.sin(yaw)-z*math.cos(yaw)])
                    end_error=float(np.linalg.norm((target-ag.state.position)[[0,2]]))
                    # Same legacy source-depth mask used in the earlier pose experiment.
                    oldsafe=features(plan,SOURCE_H,0.,np.array([0.,0.,1.]),[.5,.5,.1,.1,1.],d)[1]
                    seq=np.zeros(108,np.int8);seq[:tap]=plan
                    pad=np.zeros((108,len(NAMES)),np.float16);pad[:tap]=ph
                    rows.append(dict(group=f"{h}/{idx}/{tap}",source_index=idx,
                        kind="natural" if kind=="none" else "perturbed",perturbation=kind,
                        action=seq,path=pad,safe_mask=oldsafe,first_blocked=blocked,
                        reach=int(blocked==0 and end_error<.14),end_error_m=end_error))
            if len(used)>=ns:break
    finally:sim.close()
    assert rows
    arr={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    arr.update(split=np.repeat(split,len(rows)),house=np.repeat(h,len(rows)),path_names=np.asarray(NAMES))
    tmp=path.with_suffix(".partial.npz");np.savez_compressed(tmp,**arr);tmp.replace(path)
    print("natural",split,h,len(used),len(rows),int(arr["reach"].sum()),flush=True)


if __name__=="__main__":
    os.umask(0o077)
    p=argparse.ArgumentParser()
    p.add_argument("--freeze",action="store_true")
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--worker",type=int,default=0)
    args=p.parse_args()
    if args.freeze:freeze()
    else:
        cfg=json.loads(MANIFEST.read_text())
        jobs=[(s,h) for s,hh in cfg["cohorts"].items() for h in hh]
        for s,h in jobs[args.worker::args.workers]:run(s,h,cfg["starts"])
