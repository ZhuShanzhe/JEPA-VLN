"""Follow existing authoritative acceptances, quarantine replay problems, score no policy."""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE=Path(__file__).resolve().parent
RT=HERE.parents[1]
def load(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, obj):
    tmp=p.with_suffix(".tmp."+str(os.getpid()))
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    tmp.replace(p)


def counts(items):
    s=collections.Counter()
    for v in items:
        s["episodes"]+=1
        s["eligible"]+=v["eligible_instances"]
        s["fully_reference_validated_episodes"]+=int(not v["reference_control_missing"])
        s["reference_control_insufficient_instances"]+=v["reference_control_missing"]
        s["never_seen_unconfirmed_instances"]+=v["never_seen_unconfirmed"]
        unconfirmed = sum(t["exposure"]["moderate"]["first_qualifying_step"] is None and
                           v["heldout_reference_semantic_pixels"][str(t["id"])]<10 for t in v["targets"])
        exposure=v["count_exposed"]
        for level,k in exposure.items():
            s["instances_"+level]+=k
        k=v["k"]
        if v["reported_distinct"]<k:
            s["incomplete_episodes"]+=1
            if exposure["moderate"]>=k:
                s["incomplete_with_moderate_exposure_meeting_k"]+=1
            elif exposure["moderate"]+unconfirmed < k:
                s["incomplete_below_k_even_allowing_unconfirmed_references"]+=1
            else:
                s["incomplete_below_k_unresolved_reference_coverage"]+=1
        for w in v["candidate_windows"]:
            s["applied_multicandidate_frontier_windows"]+=1
            n=w["new_crossings"]
            if n["moderate"]:
                s["windows_first_moderate_exposure"]+=1
            elif n["any"]:
                s["windows_first_any_without_first_moderate"]+=1
            else:
                s["windows_no_new_any_or_moderate"]+=1
            if n["large"]:
                s["windows_first_large_exposure"]+=1
    return dict(s)


def main(campaign,out,once):
    os.umask(0o077)
    for n in ("verified","logs","issues"):
        (out/n).mkdir(parents=True,exist_ok=True)
    fd=(out/"worker.lock").open("a")
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock_file=campaign/"lock.json"
    lock=load(lock_file)
    fingerprint=sha(lock_file)
    panel={r["episode_id"]:r["condition"] for r in lock["episodes"]}
    assert len(panel)==300 and lock["namespace"]=="formal_frozen"
    program=HERE/"visibility_replay.py"
    script=sha(program)
    while True:
        assert sha(program)==script and sha(lock_file)==fingerprint
        accepted=sorted((campaign/"accepted/native").glob("*.json"))
        by_condition=collections.defaultdict(list)
        waited=False
        for p in accepted:
            ref=load(p)
            eid=p.stem
            assert eid in panel and ref["episode_id"]==eid and ref["namespace"]=="formal_frozen" and ref["arm"]=="native" and ref["lock"]==fingerprint
            dest=out/"verified"/(eid+".json")
            issue=out/"issues"/(eid+".json")
            if not dest.exists():
                if issue.exists() and time.time()-issue.stat().st_mtime<1800:
                    continue
                free,busy=map(int,subprocess.check_output(["nvidia-smi","--query-gpu=memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True).splitlines()[0].split(","))
                if free<4096 or busy>=85:
                    waited=True
                    continue
                source=Path(ref["result"])
                assert source.resolve()==(RT/"runs"/ref["run_id"]/"result.json").resolve() and sha(source)==ref["result_sha256"]
                svc=RT/"runs"/ref["evidence"]["server_run"]
                logpath=out/"logs"/(eid+"_"+str(time.time_ns())+".log")
                try:
                    with logpath.open("xb") as log:
                        proc=subprocess.run(["nice","-n","15",os.environ["MOST_EVAL_PYTHON"],"-B",str(program),
                                             str(source),str(svc),str(dest)],stdout=log,stderr=subprocess.STDOUT,
                                            timeout=240,cwd=os.environ["BENCH_DIR"])
                    if proc.returncode:raise RuntimeError("exit code "+str(proc.returncode))
                except (OSError,RuntimeError,subprocess.TimeoutExpired) as exc:
                    write(issue,{"code":str(exc),"log_name":logpath.name,"included":False})
                    continue
            r=load(dest)
            assert r["source_result_sha256"]==ref["result_sha256"] and r["source_script_sha256"]==script
            by_condition[panel[eid]].append(r)
        all_items=[v for vv in by_condition.values() for v in vv]
        verified=len(all_items)
        num_issues=sum(1 for p in accepted if (out/"issues"/p.name).exists() and not (out/"verified"/p.name).exists())
        report={"schema":1,"panel_sha256":lock["panel_sha256"],"replay_code_sha256":script,"accepted":len(accepted),
                "verified_episodes":verified,"quarantined_replays":num_issues,
                "all":counts(all_items),"by_condition":{k:counts(v) for k,v in by_condition.items()},
                "diagnostic":"Delivered-path visibility at 320x320. Any=1px. Moderate=103px at 2 different consecutive-view poses. Large=512px at 3. Applied frontier windows (decision_step,next_decision_step] count first-ever episode crossings, not effect of trying an alternative. Reference positives use 10px at up to 12 independently rendered annotated viewpoints."}
        write(out/"aggregate.json",report)
        if once or verified==len(panel):return
        time.sleep(30 if waited else 60)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("campaign",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--once",action="store_true")
    a=parser.parse_args()
    main(a.campaign.resolve(),a.output.resolve(),a.once)
