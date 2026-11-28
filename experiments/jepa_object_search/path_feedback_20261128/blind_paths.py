"""Apply exactly the frozen paired cue protocol in the six separate natural-check blind houses."""
import argparse
import json
import os
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent

if __name__=="__main__":
    os.umask(0o077)
    p=argparse.ArgumentParser()
    p.add_argument("--worker",type=int,default=0)
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--encode",action="store_true")
    args=p.parse_args()
    homes=json.loads((HERE/"natural_design.json").read_text())["cohorts"]["blind"][args.worker::args.workers]
    raw=HERE/"private_blind_render"
    if args.encode:
        from encode_paths import FrozenImageEncoder,encode
        import torch
        torch.set_num_threads(2)
        encoder=FrozenImageEncoder(HERE.parent)
        out=HERE/"private_blind_paired";out.mkdir(exist_ok=True)
        for house in homes:
            folder=raw/("blind_"+house)
            until=time.monotonic()+600
            while not (folder/"controls.npz").exists():
                if time.monotonic()>until:raise RuntimeError(folder.name)
                time.sleep(1)
            f=out/(folder.name+".npz")
            if not f.exists():encode(folder,f,encoder)
    else:
        import collect_paths
        collect_paths.RAW=raw
        for house in homes:collect_paths.collect("blind",house)
