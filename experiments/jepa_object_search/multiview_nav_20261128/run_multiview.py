"""Physically acquired RGB-D triplets and the previous frozen 300-action execution/scoring loop."""
import argparse
from collections import Counter, deque
import fcntl
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PRIOR=ROOT/"committed_inspect_20261128"
sys.path.insert(0,str(PRIOR))
import run_committed as old
from run_committed import read, digest, journal_name, FAMILY, physical
from multi_gate import MultiGate

PRIOR_RESULT="7d3c92d32245a3fdd756a34b7567581162fc8bf355de95016ac612fed460e618"
PRIOR_LOCK="6fa846e5467ee092cf158741a1d0437b02c83934e2861589d6e94e9564cadb29"


def seal(create=False):
    assert digest(PRIOR/"accepted.json")==PRIOR_RESULT and old.seal()==PRIOR_LOCK
    inputs=[PRIOR/"gate_lock.json",PRIOR/"accepted.json",ROOT/"jepa_reid_v21/detector/proposals.py",
            ROOT/"jepa_reid_v21/detector/protocol.json",ROOT/"b_verification/detector_run/weights/groundingdino_swint_ogc.pth",
            PRIOR/"commit_policy.py",PRIOR/"run_committed.py"]
    inputs.extend(HERE/k for k in ("multi_gate.py","multi_policy.py","multi_worker.py","cue_service.py",
                      "cue_render.py","cue_worker.py","run_multiview.py"))
    inputs.extend(PRIOR/"private_events"/journal_name(i) for i in range(8))
    inputs.extend(PRIOR/"private_dino/gpt_multiview"/f"{i}.{kind}.json" for i in range(4) for kind in ("dispatch","response"))
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in inputs}
    path=HERE/"gate_lock.json"
    if create and not path.exists():
        with path.open("x") as f: json.dump(hashes,f,indent=2)
    assert read(path)==hashes
    return digest(path)


def hash_plane(a):
    return sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class ScanRecorder:
    def __init__(self,sim):
        self.sim=sim
        self.frames=deque(maxlen=5)
        self.count=0
    def __getattr__(self,key):
        return getattr(self.sim,key)
    def save(self,obs,action):
        pose=tuple(physical.policy_pose(self.sim))
        self.frames.append(dict(step=self.count,action=action,rgb=np.asarray(obs["rgb"])[:,:,:3].copy(),
                                depth=np.asarray(obs["depth"]).copy(),pose=pose))
        self.count+=1
        return obs
    def get_sensor_observations(self):
        obs=self.sim.get_sensor_observations()
        if not self.frames:
            self.save(obs,"start")
        return obs
    def step(self,action):
        return self.save(self.sim.step(action),action)
    def packet(self,views,central_depth,current):
        f=list(self.frames)
        assert len(f)==5 and [x["action"] for x in f[1:]]==["turn_right","turn_left","turn_left","turn_right"]
        assert all(math.dist(x["pose"][:2], current[:2])<1e-5 for x in f)
        triplet=[f[1],f[0],f[3]]
        assert all(np.array_equal(x,v["rgb"]) for x,v in zip(views,triplet))
        assert np.array_equal(central_depth,triplet[1]["depth"])
        for a,item in zip((-30,0,30),triplet):
            error=math.remainder(item["pose"][2]-current[2]-math.radians(a),math.tau)
            assert abs(error)<1e-4,("bad scan angle",a,error)
            assert item["depth"].shape[:2]==item["rgb"].shape[:2]==(192,192)
        geometry=[(v["pose"],v["depth"]) for v in triplet]
        receipt=[dict(step=v["step"],pose=list(v["pose"]),rgb_sha256=hash_plane(v["rgb"]),
                      depth_sha256=hash_plane(v["depth"])) for v in triplet]
        return geometry,receipt


class BoundPolicy:
    def __init__(self,policy,scans):
        self.policy=policy
        self.scans=scans
        self.fast_continue=policy.fast_continue
    def pick(self,tracker,memory,grid,pose,scale,rows,views,goal,budget,channel,depth_m,**kw):
        geometry,receipt=self.scans.packet(views,depth_m,pose)
        decision,region=self.policy.pick(tracker,memory,grid,pose,scale,rows,views,goal,budget,channel,depth_m,
                                         scan_geometry=geometry,**kw)
        decision["physical_scan"]=receipt
        return decision,region


def seed(i, private):
    src=PRIOR/"private_events"/journal_name(i)
    dst=private/src.name
    initial=src.read_bytes()
    assert initial.endswith(b"\n")
    if dst.exists():
        assert dst.read_bytes().startswith(initial)
    else:
        tmp=dst.with_suffix(".tmp")
        tmp.write_bytes(initial)
        os.replace(tmp,dst)


def environment(spec):
    shard=physical.DATA/"eval/hssdov/content"/(spec["scene"]+".json.gz")
    assert digest(shard)==spec["shard_sha256"]
    with gzip.open(shard) as f:
        ep=next(e for e in json.load(f)["episodes"] if str(e["episode_id"])==str(spec["id"]))
    return ep, physical.sensor_sim(ep)


def check_scan():
    ep,sim=environment(read(FAMILY/"panel.json")[0])
    try:
        state=physical.hs.AgentState()
        state.position=np.float32(ep["start_position"])
        state.rotation=physical.quat_from_coeffs(ep["start_rotation"])
        sim.get_agent(0).set_state(state)
        scan=ScanRecorder(sim)
        center=scan.get_sensor_observations()
        right=scan.step("turn_right")["rgb"][:,:,:3].copy()
        scan.step("turn_left")
        left=scan.step("turn_left")["rgb"][:,:,:3].copy()
        scan.step("turn_right")
        _,receipt=scan.packet([right,center["rgb"][:,:,:3],left],center["depth"],physical.policy_pose(sim))
        assert [r["step"] for r in receipt]==[1,0,3]
        print("validated 3 observed RGB-D pairs and 3 measured poses in the original 4 physical turns")
    finally: sim.close()


def run_one(i,spec,channel,private,out,version):
    from multi_policy import MultiViewPolicy
    from cue_service import CueSelector
    with (private/f"{i}.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dst=out/f"{i}_multiview.json"
        if dst.exists():
            saved=read(dst)
            assert saved["steps"]==300 and saved["frozen_sha256"]==version
            return
        seed(i,private)
        cue=CueSelector(log_dir=private/f"cue_{i}")
        policy=MultiViewPolicy(MultiGate(str(i),private),cue=cue)
        ep,sim=environment(spec)
        scans=ScanRecorder(sim)
        try:
            tracepath,logpath=private/f"{i}.poses.jsonl",private/f"{i}.jsonl"
            for p in (tracepath,logpath):
                if p.exists(): p.rename(p.with_suffix(f".attempt_{time.time_ns()}.jsonl"))
            with logpath.open("x") as log, tracepath.open("x",buffering=1) as trace:
                result=old.rollout(scans,ep,spec,channel,BoundPolicy(policy,scans),log,trace)
            assert result["steps"]==spec["budget"]==300 and result["reachable_instances_at_start"]>=2
            assert scans.count==301
            counters={k:Counter() for k in ("gpt_mode","gpt_evidence","gpt_cue","exec_status")}
            for line in logpath.read_text().splitlines():
                e=json.loads(line)
                if e.get("gpt_called"):
                    assert e["old_selected_region_id"]==e["executed_region_id"] and e["old_selected_region_key"]==e["region_key"]
                    for k,c in counters.items(): c[str(e.get(k))]+=1
            result.update(arm="multiview",frozen_sha256=version,upper_decisions=counters,
                          private_trace_sha256=digest(tracepath),private_events_sha256=digest(logpath))
            tmp=dst.with_suffix(".tmp")
            tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
            os.replace(tmp,dst)
            print("complete",i,flush=True)
        finally:
            sim.close()
            cue.close()


def main():
    parser=argparse.ArgumentParser()
    op=parser.add_mutually_exclusive_group(required=True)
    op.add_argument("--seal-only",action="store_true")
    op.add_argument("--check-scan",action="store_true")
    op.add_argument("--indices",type=int,nargs="+",choices=range(8))
    args=parser.parse_args()
    os.umask(0o077)
    if args.check_scan:
        check_scan()
        return
    version=seal(args.seal_only)
    if args.seal_only:
        print(version)
        return
    from multi_worker import start_worker
    private,out=HERE/"private_events",HERE/"runs"
    private.mkdir(exist_ok=True,mode=0o700)
    out.mkdir(exist_ok=True)
    channel,worker=start_worker(ROOT,private/f"worker_{args.indices[0]}.log")
    try:
        panel=read(FAMILY/"panel.json")
        for i in args.indices: run_one(i,panel[i],channel,private,out,version)
    finally:
        channel.close()
        worker.wait(timeout=60)


if __name__=="__main__":
    main()
