"""Collect each preassigned house once; keep failures visible and all prespecified origins."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess


def main(args):
    protocol=json.loads((args.root/"cohort.json").read_text())
    jobs=[(k,h) for k in ("train","calibration","test") for h in protocol[k]]
    logs=args.root/"jobs";logs.mkdir(exist_ok=True)
    def one(job):
        split,house=job
        with (logs/(split+"-"+house+".log")).open("a") as f:
            run=subprocess.run([str(args.python),str(args.root/"collect_branches.py"),str(args.assets),str(args.root),split,house,
                                "--target","24","--trials","450"],stdout=f,stderr=f,env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                                HABITAT_SIM_LOG="quiet",MAGNUM_LOG="quiet"))
        print(split,house,run.returncode,flush=True)
        return (split,house,run.returncode)
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        results=list(pool.map(one,jobs))
    if any(q[2] for q in results):raise RuntimeError("house job failures")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("root","python","assets"):p.add_argument(k,type=Path)
    p.add_argument("--workers",type=int,default=5)
    main(p.parse_args())
