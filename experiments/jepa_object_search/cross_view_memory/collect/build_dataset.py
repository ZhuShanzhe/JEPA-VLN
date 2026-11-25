"""Collect the preassigned houses and validate the finished private manifest."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image


def run(a):
    frozen = json.loads(a.cohort.read_text())
    houses = {"train":frozen["train"], "val":frozen["calibration"], "test":frozen["test"]}
    prefixes = [h.split("_")[0] for hs in houses.values() for h in hs]
    assert len(prefixes)==len(set(prefixes)), "house family overlaps"
    a.dest.mkdir(parents=True,exist_ok=True)
    (a.dest/"jobs").mkdir(exist_ok=True)
    config = dict(houses=houses, source_hash=hashlib.sha256(a.cohort.read_bytes()).hexdigest(),
                  collector_sha256=hashlib.sha256(Path(__file__).with_name("collect_views.py").read_bytes()).hexdigest(),
                  object_padding_each_side=.15, context_padding_each_side=.8,
                  min_target_pixels=170,min_angle_deg=15,min_camera_distance=.35)
    frozen_path = a.dest/"sampling_protocol.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != config:
        raise RuntimeError("private output already contains a different cohort or collector")
    frozen_path.write_text(json.dumps(config,indent=2))
    def worker(job):
        split, house = job
        log = a.dest/"jobs"/f"{split}_{house}.log"
        with log.open("a") as f:
            res = subprocess.run([sys.executable,str(Path(__file__).with_name("collect_views.py")),
                                  str(a.assets),str(a.dest),split,house],stdout=f,stderr=f,
                                 env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                                          HABITAT_SIM_LOG="quiet",MAGNUM_LOG="quiet"))
        print(split,house,res.returncode,flush=True)
        return res.returncode
    jobs = [(s,h) for s,hs in houses.items() for h in hs]
    with ThreadPoolExecutor(a.workers) as pool:
        codes = list(pool.map(worker,jobs))
    if any(codes):
        raise RuntimeError("unfinished house; see per-house logs")
    rows=[]
    stats={}
    for split, hs in houses.items():
        subset=[]
        for h in hs:
            house_rows=json.loads((a.dest/"houses"/f"{split}_{h}.json").read_text())
            assert all(r["house"]==h and r["split"]==split for r in house_rows)
            subset.extend(house_rows)
        inst=Counter(r["instance"] for r in subset)
        cats=defaultdict(set)
        angles=defaultdict(list)
        for r in subset:
            cats[r["house"],r["category"]].add(r["instance"])
            angles[r["instance"]].append(r["yaw_deg"])
            for field in ("image","context_image"):
                with Image.open(a.dest/r[field]) as im:
                    im.verify()
        assert min(inst.values())>=2 and min(map(len,cats.values()))>=2
        stats[split]=dict(assigned_houses=len(hs),houses=len({r["house"] for r in subset}),instances=len(inst),
                         views=len(subset),categories=dict(Counter(r["category"] for r in subset)),
                         assets=len({r["asset"] for r in subset}),angle_pairs_more_45=sum(
                             abs((a-b+180)%360-180)>=45
                             for values in angles.values() for a,b in combinations(values,2)))
        rows.extend(subset)
    assert len({r["image"] for r in rows})==len(rows)
    manifest=a.dest/"manifest.jsonl"
    temp=manifest.with_suffix(".tmp")
    temp.write_text("".join(json.dumps(r,separators=(",",":"))+"\n" for r in rows))
    sig=hashlib.sha256(temp.read_bytes()).hexdigest()
    report=dict(splits=stats,manifest_sha256=sig)
    (a.dest/"audit.json").write_text(json.dumps(report,indent=2))
    temp.replace(manifest)
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=ArgumentParser()
    for key in ("assets","cohort","dest"):
        p.add_argument(key,type=Path)
    p.add_argument("--workers",type=int,default=4)
    run(p.parse_args())
