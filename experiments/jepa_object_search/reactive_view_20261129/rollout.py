"""Continuous physical rollouts from the frozen new-view benchmark. No reset after contact."""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"new_view_20261129"
sys.path.insert(0,str(OLD))
from sample_views import ACTION, ASSETS, SOURCE_H, camera, inventory, simulator


class Worker:
    def __init__(self, folder, policy):
        self.packet=folder/"route_inputs.npz"
        self.log=(folder/"inference.log").open("a")
        py=os.environ.get("JEPA_POLICY_PYTHON","/root/autodl-tmp/jepa_instance_bench_20261016/policy_env/bin/python")
        env=dict(os.environ)
        env.pop("PYTHONPATH",None)
        env.pop("LD_LIBRARY_PATH",None)
        self.proc=subprocess.Popen([py,"-u",str(HERE/"route_worker.py"),str(self.packet),policy],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,env=env)

    def choose(self,ctx,box,history,left):
        np.savez(self.packet,context=ctx,box=box,pose=[f["pose"] for f in history],
                 bump=[f["bump"] for f in history],rgb=[f["rgb"] for f in history],
                 depth=[f["depth"] for f in history],remaining=left)
        self.proc.stdin.write("predict\n");self.proc.stdin.flush()
        line=self.proc.stdout.readline()
        if not line:raise RuntimeError("route worker exited; check its log")
        d=json.loads(line)
        if "error" in d:raise RuntimeError(d["error"])
        return d

    def close(self):
        try:self.proc.stdin.close()
        finally:
            self.proc.wait(timeout=60)
            self.log.close()
            self.packet.unlink(missing_ok=True)


def audit_box(mask,box):
    pixels=int(mask.sum())
    if box is None or not pixels:return 0,0.0
    x1,y1,x2,y2=box
    yy,xx=np.where(mask);l,t,r,b=int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1
    intersect=max(0,min(x2,r)-max(x1,l))*max(0,min(y2,b)-max(y1,t))
    union=(r-l)*(b-t)+(x2-x1)*(y2-y1)-intersect
    return int(mask[y1:y2,x1:x2].sum()),intersect/max(1,union)


def one(sim,worker,store,src_dir,d,ctx,row,mode,motion=None):
    house,source=row["house"],int(row["source"])
    rows=np.flatnonzero(d["group"]==row["group"])
    budget=int(d["budget"][rows[0]])
    with np.load(src_dir/f"{source}_sensor.npz") as s:
        start,yaw,oid,box=s["position"],float(s["yaw"]),int(s["oid"]),s["box"].tolist()
    ag=sim.get_agent(0)
    obs=camera(sim,start,yaw,SOURCE_H,0.)
    confirmer=None
    if mode!="route":
        from frame_confirm import FrameConfirmer
        confirmer=FrameConfirmer(obs["rgb"][...,:3].copy(),obs["depth"].squeeze().copy(),box)
    base_route=row["jepa"]
    plan=tuple(map(int,d["action"][base_route][d["action"][base_route]>0])) if base_route is not None else ()
    history=[];events=[];replans=[]
    first=None
    def append(obs,action,bump):
        nonlocal first
        delta=ag.state.position-start
        q=ag.state.rotation
        # The deployed base has yaw-only rotations and two translation axes.
        world_yaw=2*math.atan2(q.y,q.w)
        heading=math.atan2(math.sin(world_yaw-yaw),math.cos(world_yaw-yaw))
        pose=(math.cos(yaw)*delta[0]-math.sin(yaw)*delta[2],
              -math.sin(yaw)*delta[0]-math.cos(yaw)*delta[2],heading)
        rgb=obs["rgb"][...,:3].copy()
        depth=obs["depth"].squeeze().copy()
        history.append(dict(pose=pose,bump=bool(bump),rgb=rgb,depth=depth))
        # The confirmation estimator receives no environment labels at any step.
        verify=confirmer.observe(rgb,depth,pose) if first is None and confirmer is not None else {}
        accepted=bool(verify.get("accept",False))
        bb=verify.get("box")
        if bb is not None:
            x,y,r,b=map(int,bb);bb=[max(0,x),max(0,y),min(rgb.shape[1],r),min(rgb.shape[0],b)]
            if bb[0]>=bb[2] or bb[1]>=bb[3]:bb=None
        if accepted and bb is None:raise RuntimeError("invalid confirmation rectangle")
        # Evaluation-only branch after the action decision. Never return any labels to the worker.
        mask=obs["ids"]==oid
        pixels=int(mask.sum())
        inter,iou=audit_box(mask,bb)
        t=len(events)
        if accepted:
            first=t;name=f"{source}_step{t}"
            Image.fromarray(rgb).save(store/(name+".png"))
            l,y,r,b=bb
            Image.fromarray(rgb[y:b,l:r]).save(store/(name+"_crop.png"))
        events.append(dict(t=t,action=int(action),bump=bool(bump),pose=list(map(float,pose)),px=pixels,
            accept=accepted,box=bb,intersection=inter,gt_box_iou=float(iou),
            correct=bool(accepted and pixels>=369 and inter>=.5*pixels and iou>=.30),
            p_same=float(verify.get("p_same",0)),p_clear=float(verify.get("p_clear",0)),
            estimated_pixels=float(verify.get("estimated_pixels",0))))
    def act(a):
        previous=ag.state.position.copy()
        hit=ag.act(ACTION[a])
        stall=a==1 and np.linalg.norm((ag.state.position-previous)[[0,2]])<.23
        bump=bool(a==1 and (hit or stall))
        append(sim.get_sensor_observations(),a,bump)
        return bump
    append(obs,0,False)
    for a in (2,3):
        assert not act(a)
    bumped=False
    for a in plan:
        bumped=act(a)
        if bumped:break
    base_len=len(events)-1
    if base_route is not None:
        assert (base_len-2 if bumped else 0)==d["first_blocked"][base_route]
        if not bumped:assert abs(events[-1]["px"]-d["pixels"][base_route])<=1
    # After the first confirmation, further motion is a counterfactual no-stop arm only.
    # This preserves exact route-only controls with identical history up to first acceptance.
    while len(events)-1<budget:
        t=len(events)-1
        if motion is None:
            decision=worker.choose(ctx,box,history,budget-t)
        else:
            saved=motion["replans"][len(replans)]
            assert t==saved["t"]
            decision=dict(**saved["details"],actions=saved["actions"])
        new=list(map(int,decision["actions"]))
        if len(new)>budget-t or not all(a in (1,2,3) for a in new):
            raise ValueError("invalid or over-budget recovery")
        safe={k:v for k,v in decision.items() if k in ("reason","mode","score","p_reach",
              "p_visible","range_m","bearing_deg","x","z","yaw","target","exposure","view_id",
              "p_visible_if_reached","n_candidates","n_admitted","candidates","admitted",
              "local_distance_m","relative_bearing_deg","candidate_views","admitted_views","map_keyframes","contact_edges")}
        replans.append(dict(t=t,trigger="contact_or_stall" if bumped else "end_of_plan",
                            actions=new,details=safe))
        if not new:break
        bumped=False
        for a in new:
            bumped=act(a)
            if bumped:break
    assert len(events)-1<=budget
    if motion is not None:
        assert len(events)==len(motion["events"]) and base_len==motion["baseline_length"]
        for event,src in zip(events,motion["events"]):
            assert event["action"]==src["action"] and event["bump"]==src["bump"]
            assert abs(event["px"]-src["px"])<=1 and np.allclose(event["pose"],src["pose"],atol=1e-5)
    old_confirm=first if first is not None and first<=base_len else None
    g=d["kind"][rows]=="old"
    seven_blocked=not d["reach"][rows[g]].any()
    focus=bool(seven_blocked and max(events[t]["px"] for t in (0,1,2))<369)
    return dict(house=house,source=source,group=row["group"],budget=budget,focus=focus,
                seven_blocked=seven_blocked,baseline_length=base_len,old_confirmation=old_confirm,
                new_confirmation=first,events=events,replans=replans)


def main(house_index,mode,policy):
    os.umask(0o077)
    frozen=json.loads((OLD/"view_design.json").read_text())["cohorts"]["blind"]
    house=frozen[house_index]
    src_dir=OLD/"private_render_blind60"/("blind_"+house)
    suffix="_area" if policy=="projected_area" else ""
    store=HERE/(("private_motion" if mode=="route" else "private_roll")+suffix)/house
    store.mkdir(parents=True,exist_ok=True)
    choices=json.loads((OLD/"private_evaluation/blind_choices.json").read_text())
    with np.load(src_dir/"controls.npz") as f:
        d={k:f[k] for k in ("group","kind","source_index","budget","action","pixels","reach","first_blocked")}
    with np.load(OLD/"private_data"/("blind_"+house+".npz")) as f:ctx=f["context"]
    sim=simulator(house)
    worker=Worker(store,policy) if mode!="confirm" else None
    try:
        inventory(sim,ASSETS)
        for row in (r for r in choices if r["house"]==house):
            target=store/f"{row['source']}.json"
            if target.exists():continue
            index=int(np.flatnonzero(d["group"]==row["group"])[0])
            motion=json.loads((HERE/("private_motion"+suffix)/house/target.name).read_text()) if mode=="confirm" else None
            out=one(sim,worker,store,src_dir,d,ctx[index],row,mode,motion)
            out["recovery_model"]=policy
            temp=target.with_suffix(".partial");temp.write_text(json.dumps(out,allow_nan=False));temp.replace(target)
            if int(row["source"])%10==0:print(house,row["source"],flush=True)
    finally:
        if worker:worker.close()
        sim.close()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("house_index",type=int)
    p.add_argument("--mode",choices=("route","confirm","live"),default="live")
    p.add_argument("--policy",choices=("jepa","projected_area"),default="jepa")
    args=p.parse_args();main(args.house_index,args.mode,args.policy)
