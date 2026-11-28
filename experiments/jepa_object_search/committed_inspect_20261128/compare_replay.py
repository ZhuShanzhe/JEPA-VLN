"""Replay the historical inspect-only decisions against exact observation hashes, offline."""
import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

from run_committed import HERE, ROOT, FAMILY, PRIOR, PRIOR_RESULTS, TracedSim, digest, journal_name, physical, read
from trigger_policy import InspectOnlyPolicy
import gpt_gate as contract
from audit_committed import METRICS, lines, trajectory

GATE_SHA="eea2ecf8bc0bbb949b734c28d73dea3966f361fa9646429cf416002c1848932b"
PRIVATE=HERE/"private_previous"


class ReplayOnly:
    def __init__(self, path):
        self.returned = []
        self.responses = {}
        for e in lines(path):
            if e["event"]=="success":
                assert e["model"]==contract.MODEL and e["hash"] not in self.responses
                self.responses[e["hash"]]=e["decision"]

    def choose(self, views, goal, regions, patches, memory):
        summary=contract.clean(dict(goal=goal,region_columns=contract.REGION_FIELDS,
                           regions=[[r.get(k) for k in contract.REGION_FIELDS] for r in regions],
                           patch_columns=contract.PATCH_FIELDS,
                           patches=[[p.get(k) for k in contract.PATCH_FIELDS] for p in patches],memory=memory))
        image_sha=contract.hashlib.sha256(contract.strip_png(views)).hexdigest()
        content=[contract.MODEL,contract.SYSTEM,contract.encoded(summary).decode(),image_sha,"png-high",0,160]
        key=contract.hashlib.sha256(contract.encoded(content)).hexdigest()
        if key not in self.responses:
            raise RuntimeError("unseen request in offline-only historical replay; abort")
        response=contract.decision(self.responses[key],{r["id"] for r in regions},{p["patch_id"] for p in patches})
        self.returned.append(key)
        return response


def replay(i,spec,channel,release):
    result=PRIVATE/f"{i}.accepted.json"
    with (PRIVATE/f"{i}.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rowpath=PRIOR/"runs"/f"{i}_inspect.json"
        assert release["by_house"][i]["verified_result_sha256"]==digest(rowpath)
        control=read(rowpath)
        tracepath,logpath=PRIVATE/f"{i}.poses.jsonl",PRIVATE/f"{i}.jsonl"
        if result.exists():
            check=read(result)
            assert check["prior_run_sha256"]==digest(rowpath) and check["poses_sha256"]==digest(tracepath)
            assert check["replay_events_sha256"]==digest(logpath)
            return
        prior_receipts=PRIOR/"private_events"/journal_name(i)
        prior_events=PRIOR/"private_events"/f"{i}.jsonl"
        receipts=lines(prior_receipts)
        gate=ReplayOnly(prior_receipts)
        policy=InspectOnlyPolicy(gate)
        shard=physical.DATA/"eval/hssdov/content"/(spec["scene"]+".json.gz")
        assert digest(shard)==spec["shard_sha256"]
        with gzip.open(shard) as f:
            episode=next(e for e in json.load(f)["episodes"] if str(e["episode_id"])==str(spec["id"]))
        for path in (tracepath,logpath):
            if path.exists():
                path.rename(path.with_suffix(f".attempt_{time.time_ns()}.jsonl"))
        sim=physical.sensor_sim(episode)
        try:
            with logpath.open("x") as log,tracepath.open("x",buffering=1) as trace:
                traced=TracedSim(sim,trace)
                with patch.object(physical,"pick",policy.pick),patch.object(contract.http.client,"HTTPConnection",
                            side_effect=RuntimeError("HTTP dispatch disabled in offline replay")):
                    row=physical.episode_run(traced,episode,spec,channel,log)
        finally:
            sim.close()
        assert row["steps"]==300 and row["arm"]=="local"
        for field in METRICS+("first_steps","scene","episode","goal","stats","counts","distance_by_phase",
                              "instances_available","reachable_instances_at_start"):
            assert row[field]==control[field],field
        old_events=lines(prior_events)
        new_events=lines(logpath)
        assert new_events==old_events,"non-identical historical decisions"
        assert gate.returned==[e["hash"] for e in receipts if e["event"] in ("success","cache")][-len(gate.returned):]
        trajectory(lines(tracepath))
        metadata=dict(index=i,goal=spec["goal"],scene=spec["scene"],episode=spec["id"],
                      steps=300,matched_full_request_hashes=len(gate.returned),matched_events=len(old_events),
                      prior_run_sha256=digest(rowpath),poses_sha256=digest(tracepath),replay_events_sha256=digest(logpath),
                      fresh_requests=0,matched_metrics={key:row[key] for key in METRICS+("first_steps",)})
        tmp=result.with_suffix(".tmp")
        tmp.write_text(json.dumps(metadata,indent=2)+"\n")
        os.replace(tmp,result)
        print("historical replay verified",i,flush=True)


def main():
    parser=argparse.ArgumentParser()
    operation=parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--indices",type=int,nargs="+",choices=range(8))
    operation.add_argument("--audit",action="store_true")
    args=parser.parse_args()
    os.umask(0o077)
    assert digest(Path(contract.__file__))==GATE_SHA and digest(PRIOR/"accepted.json")==PRIOR_RESULTS
    release=read(PRIOR/"accepted.json")
    assert digest(FAMILY/"panel.json")==release["panel_sha256"]
    for file in (FAMILY/"local_lock.json",PRIOR/"gate_lock.json"):
        for rel,h in read(file).items():
            assert digest(ROOT/rel)==h,rel
    PRIVATE.mkdir(mode=0o700,exist_ok=True)
    if args.audit:
        results=[read(PRIVATE/f"{i}.accepted.json") for i in range(8)]
        for i,r in enumerate(results):
            assert r["index"]==i and r["steps"]==300 and r["fresh_requests"]==0
            assert digest(PRIVATE/f"{i}.poses.jsonl")==r["poses_sha256"]
            assert digest(PRIVATE/f"{i}.jsonl")==r["replay_events_sha256"]
            assert r["prior_run_sha256"]==release["by_house"][i]["verified_result_sha256"]
        report=dict(houses=len(results),actions=sum(r["steps"] for r in results),
                    matched_requests=sum(r["matched_full_request_hashes"] for r in results),fresh_requests=0,
                    decision_sequences_all_identical=True,
                    paired_first_hit_steps_all_identical=True,
                    scene_count=len({r["scene"] for r in results}),source_sha256=digest(Path(__file__)))
        (HERE/"previous_replay.json").write_text(json.dumps(report,indent=2)+"\n")
        print(report)
        return
    channel,worker=physical.start_worker(ROOT,PRIVATE/f"worker_{args.indices[0]}.log")
    try:
        panel=read(FAMILY/"panel.json")
        for i in args.indices:
            replay(i,panel[i],channel,release)
    finally:
        channel.close()
        worker.wait(timeout=60)


if __name__=="__main__":
    main()
