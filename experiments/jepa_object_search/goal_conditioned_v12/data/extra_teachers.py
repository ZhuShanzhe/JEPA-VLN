"""Produce first-discovery teachers for newly collected training houses only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(project,hab,assets):
    import torch
    from region_features import encode_file,grid_overlaps,summarize
    sys.path.insert(0,str(project/"jepa_hidden_v11/models"))
    from encode import Encoder
    raw=project/"jepa_route_v12/expansion/hssd"
    data=project/"jepa_route_v12/data"
    cohort=json.loads((raw/"cohort.json").read_text())
    houses=cohort["train"]
    assert not set(houses)&set(cohort["test"])
    mask_root=data/"region_masks/extra_train"
    dest=data/"local_targets/extra_train"
    dest.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    enc=Encoder(project);area=grid_overlaps()
    def extract(h):
        with (data/"logs"/("extra-"+h+".log")).open("a") as f:
            subprocess.run([str(hab),"-Bu",str(data/"region_labels.py"),str(project),str(assets),"--split","extra_train",
                            "--raw-parent",str(raw/"train"),"--house",h],stdout=f,stderr=subprocess.STDOUT,check=True,
                           env=dict(os.environ,MAGNUM_LOG="quiet",HABITAT_SIM_LOG="quiet",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"))
    remaining=set(houses); workers={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        while remaining or workers:
            for h in list(workers):
                if workers[h].done():
                    workers.pop(h).result()
                    for f in sorted((mask_root/h).glob("*.npz")):
                        if not (dest/f.name).is_file():
                            encode_file(enc,area,f,raw/"train"/h/f.name,dest/f.name)
                    print(h,flush=True)
            for h in sorted(remaining):
                receipt=raw/"train"/h/"receipt.json"
                if len(workers)>=2:break
                if receipt.exists() and json.loads(receipt.read_text())["maximum_trials"]==450:
                    remaining.remove(h);workers[h]=pool.submit(extract,h)
            if remaining or workers:time.sleep(2)
    stats=summarize(dest)
    expected=sum(json.loads((raw/"train"/h/"receipt.json").read_text())["origins"] for h in houses)
    assert stats["counts"]["origins"]==expected
    (data/"stats_extra_train.json").write_text(json.dumps(stats,indent=2))
    print(stats["counts"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("project","hab","assets"):parser.add_argument(name,type=Path)
    args=parser.parse_args();main(args.project,args.hab,args.assets)
