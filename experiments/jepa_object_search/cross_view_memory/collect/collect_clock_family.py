"""Supplemental transfer cohort of five HSSD clock synsets. No model fitting."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

import collect_views as v


CLOCKS = {"clock","wall_clock","mantel_clock","alarm_clock","grandfather_clock"}


def house_job(a, h):
    target=a.dest/"houses"/f"test_{h}.json"
    if target.exists():
        return
    seed=int(hashlib.sha256(("clock-supplement-v20|"+h).encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed)
    sim=v.open_sim(a.assets,h)
    rows=[]
    try:
        v.WORDS.update(CLOCKS)
        groups=v.inventory(sim,a.assets)
        candidates=sorted((sid,asset,center,c) for c in CLOCKS for sid,asset,center in groups[c])
        if len(candidates)>1:
            for sid,asset,center,subtype in candidates:
                views=v.sample_object(sim,h,"test",sid,asset,center,"clock",a.dest,rng)
                if len(views)>=2:
                    for r in views:
                        r["subtype"]=subtype
                    rows.extend(views)
    finally:
        sim.close()
    if len({r["instance"] for r in rows})<2:
        rows=[]
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(rows))


def main(a):
    if a.house:
        return house_job(a,a.house)
    houses=json.loads((a.primary/"sampling_protocol.json").read_text())["houses"]["test"]
    a.dest.mkdir(parents=True,exist_ok=True)
    logs=a.dest/"jobs"
    logs.mkdir(exist_ok=True)
    def job(h):
        with (logs/f"{h}.log").open("a") as f:
            ret=subprocess.run([sys.executable,__file__,str(a.assets),str(a.primary),str(a.dest),"--house",h],
                               stdout=f,stderr=f,env=dict(os.environ,MAGNUM_LOG="quiet",HABITAT_SIM_LOG="quiet",
                                                          OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"))
        return ret.returncode
    with ThreadPoolExecutor(4) as pool:
        if any(pool.map(job,houses)):
            raise RuntimeError("clock collection incomplete")
    rows=[r for h in houses for r in json.loads((a.dest/"houses"/f"test_{h}.json").read_text())]
    homes=Counter(r["house"] for r in rows)
    for r in rows:
        for k in ("image","context_image"):
            with Image.open(a.dest/r[k]) as im:
                im.verify()
    data="".join(json.dumps(r,separators=(",",":"))+"\n" for r in rows)
    digest=hashlib.sha256(data.encode()).hexdigest()
    audit=dict(eligible_house_definition="at least two distinct instances with at least two valid views",
               requested_houses=len(houses),eligible_houses=len(homes),instance_count=len({r["instance"] for r in rows}),
               view_count=len(rows),subtype_views=Counter(r["subtype"] for r in rows),sha256=digest,
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               core_sampler_sha256=hashlib.sha256(Path(v.__file__).read_bytes()).hexdigest())
    (a.dest/"audit.json").write_text(json.dumps(audit,indent=2))
    temp=a.dest/"manifest.tmp"
    temp.write_text(data)
    temp.replace(a.dest/"manifest.jsonl")
    print(json.dumps(audit),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("assets","primary","dest"):
        p.add_argument(k,type=Path)
    p.add_argument("--house")
    main(p.parse_args())
