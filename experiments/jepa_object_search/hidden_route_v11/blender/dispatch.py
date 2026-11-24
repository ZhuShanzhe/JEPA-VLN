"""Dispatch distinct house-level Habitat jobs; a valid receipt marks a finished house."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys

root, python = Path(sys.argv[1]).resolve(), sys.argv[2]


def collect(i):
    house = f"tri11_{i:02}"
    split = "train" if i <= 8 else ("calibration" if i<=10 else "test")
    if (root/split/house/"receipt.json").exists():
        return i,0
    with (root/"logs"/(house+".log")).open("a") as log:
        proc = subprocess.run([python, "-Bu", str(Path(__file__).with_name("collect.py")), str(root), str(i)],
            env=dict(os.environ, OMP_NUM_THREADS="1",MAGNUM_LOG="quiet",HABITAT_SIM_LOG="quiet"),stdout=log,stderr=log)
    return i,proc.returncode


if __name__=="__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        codes=list(pool.map(collect,range(1,17)))
    print(codes,flush=True)
    if any(code for _,code in codes):
        sys.exit(2)
