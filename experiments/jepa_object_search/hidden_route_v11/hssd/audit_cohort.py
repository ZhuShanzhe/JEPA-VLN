import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time


def run(args):
    cohort=json.loads((args.root/"cohort.json").read_bytes())
    tasks=[(s,h) for s in ("train","calibration","test") for h in cohort[s]]
    def check(job):
        s,h=job;receipt=args.root/s/h/"receipt.json";deadline=time.monotonic()+3600
        if (args.root/"physical_checks"/(s+"-"+h+".json")).exists():return
        while not receipt.exists() or json.loads(receipt.read_bytes())["maximum_trials"]!=450:
            if time.monotonic()>deadline:raise RuntimeError("collection incomplete")
            time.sleep(3)
        log=args.root/"jobs"/("audit-"+s+"-"+h+".log")
        with log.open("a") as f:
            subprocess.run([str(args.python),str(args.root/"verify_physical.py"),str(args.assets),
                            str(args.root),s,h],stdout=f,stderr=f,check=True,
                           env=dict(os.environ,HABITAT_SIM_LOG="quiet",MAGNUM_LOG="quiet",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"))
    with ThreadPoolExecutor(args.workers) as pool:list(pool.map(check,tasks))
    print("all geometry/observation replays passed",len(tasks),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("root","python","assets"):p.add_argument(k,type=Path)
    p.add_argument("--workers",type=int,default=4);run(p.parse_args())
