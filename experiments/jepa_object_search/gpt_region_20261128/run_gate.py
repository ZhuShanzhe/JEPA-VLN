"""Run the GPT upper policy on the frozen paired test, preserving the physical harness."""
import argparse
import collections
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT/"local_target_20261128"
STAGE=OLD/"eval_family"
sys.path.insert(0,str(OLD/"eval"))
import run_local as physical
from arm_policy import GatedPolicy
from gpt_gate import GPTGate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal():
    old=json.loads((STAGE/"local_lock.json").read_text())
    for name,sha in old.items():
        assert digest(ROOT/name)==sha,name
    paths=[STAGE/"panel.json",STAGE/"local_lock.json",OLD/"eval/run_local.py",
           HERE/"arm_policy.py",HERE/"gpt_gate.py",HERE/"run_gate.py"]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    assert hashes[str((STAGE/"panel.json").relative_to(ROOT))]=="9d3167d378f2f31fcd7df78d95e222fe150028bcfff739fea0f9334015e69674"
    lock=HERE/"gate_lock.json"
    if lock.exists():assert json.loads(lock.read_text())==hashes
    else:
        with lock.open("x") as f:json.dump(hashes,f,indent=2)
    return digest(lock)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--indices",type=int,nargs="+")
    parser.add_argument("--seal-only",action="store_true")
    args=parser.parse_args()
    if not args.seal_only:assert (HERE/"gate_lock.json").exists()
    version=seal()
    if args.seal_only:
        print(version)
        return
    os.umask(0o077)
    panel=json.loads((STAGE/"panel.json").read_text())
    private=HERE/"private_events"
    private.mkdir(exist_ok=True,mode=0o700)
    out=HERE/"runs"
    out.mkdir(exist_ok=True)
    channel,process=physical.start_worker(ROOT,private/f"worker_{args.indices[0]}.log")
    try:
        for i in args.indices:
            with (private/f"{i}.lock").open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                dst=out/f"{i}_gpt.json"
                spec=panel[i]
                if dst.exists():
                    prior=json.loads(dst.read_text())
                    assert prior["frozen_sha256"]==version
                    continue
                shard=physical.DATA/"eval/hssdov/content"/(spec["scene"]+".json.gz")
                assert digest(shard)==spec["shard_sha256"]
                ep=next(e for e in json.load(gzip.open(shard))["episodes"] if str(e["episode_id"])==str(spec["id"]))
                gate=GPTGate(str(i),private)
                policy=GatedPolicy(gate)
                sim=physical.sensor_sim(ep)
                try:
                    log_path=private/f"{i}.jsonl"
                    if log_path.exists():log_path.rename(private/f"{i}.attempt_{time.time_ns()}.jsonl")
                    # Replace only the high-level policy call in the frozen driver.
                    with log_path.open("x") as log, patch.object(physical,"pick",policy.pick):
                        row=physical.episode_run(sim,ep,spec,channel,log)
                    assert row["steps"]==spec["budget"]==300 and row["reachable_instances_at_start"]>=2
                    row.update(arm="gpt_region",frozen_sha256=version)
                    counters={k:collections.Counter() for k in ("gpt_mode","gpt_evidence","gpt_cue","exec_status")}
                    for line in log_path.read_text().splitlines():
                        event=json.loads(line)
                        if event.get("gpt_called"):
                            for k,c in counters.items():c[str(event.get(k))]+=1
                    row["upper_decisions"]={k:dict(c) for k,c in counters.items()}
                    tmp=dst.with_suffix(".tmp")
                    tmp.write_text(json.dumps(row,indent=2)+"\n")
                    tmp.replace(dst)
                    print("complete",i,flush=True)
                finally:sim.close()
    finally:
        channel.close()
        process.wait(timeout=60)


if __name__=="__main__":main()
