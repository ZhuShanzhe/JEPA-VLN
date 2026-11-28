"""Encode real current RGB, ideal cue crop and physical optical teachers, retaining house splits."""
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from long_data import FrozenImageEncoder
from pose_data_contract import FEATURES

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RAW=HERE/"private_pose_render"
OUT=HERE/"private_pose"


def encode(house_dir,encoder):
    rows=json.loads((house_dir/"samples.json").read_text())
    split,house=house_dir.name.split("_",1)
    assert rows
    names=sorted({r[k] for r in rows for k in ("source","cue","future") if r[k]})
    feats={"":np.zeros(1024,np.float16)}
    for start in range(0,len(names),64):
        images=[]
        for n in names[start:start+64]:
            with Image.open(house_dir/n) as im: images.append(np.asarray(im.convert("RGB")))
        for name,emb in zip(names[start:start+64],encoder.encode(images)):
            feats[name]=emb
    arr={k:np.asarray([r[k] for r in rows]) for k in (
        "group","category","plan_id","pose","reach","visible","at_position","pixels","safe_mask",
        "height","pitch_deg","native_mask","source_point_count","in_fov","intercepted")}
    plan=np.zeros((len(rows),12),np.int8)
    for i,r in enumerate(rows):plan[i,:len(r["action"])]=r["action"]
    arr.update(action=plan, future_valid=arr["visible"]>=0,
               context=np.asarray([np.concatenate([feats[r["source"]],feats[r["cue"]]]) for r in rows]),
               future=np.asarray([feats[r["future"]] for r in rows]),pose=arr["pose"].astype(np.float32),
               source_height_m=np.full(len(rows),1.31,np.float32),
               feature_names=np.asarray(FEATURES),house=np.repeat(house,len(rows)),
               split=np.repeat(split,len(rows)))
    assert not np.any((arr["reach"]==1)&~arr["future_valid"])
    assert np.all(arr["intercepted"]<=arr["in_fov"])
    assert len(set(arr["group"]))==len(rows)//81
    tmp=OUT/(house_dir.name+".pending")
    with tmp.open("wb") as f:np.savez_compressed(f,**arr)
    tmp.replace(OUT/(house_dir.name+".npz"))
    print("encoded",house_dir.name,len(rows),flush=True)


def main():
    os.umask(0o077)
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    encoder=FrozenImageEncoder(ROOT)
    cohort=ROOT/"jepa_env_v13/simulator"
    splits=json.loads((cohort/"cohort.json").read_text())
    houses={"train":json.loads((cohort/"training_cohort.json").read_text())["train"],
            "val":splits["calibration"],"test":splits["fresh"]}
    remaining={s+"_"+h for s,hh in houses.items() for h in hh}
    last=time.monotonic()
    while remaining:
        progressed=False
        for name in sorted(remaining.copy()):
            if not (RAW/name/"samples.json").exists():continue
            if not (OUT/(name+".npz")).exists():encode(RAW/name,encoder)
            remaining.remove(name)
            progressed=True
        if not progressed:
            if time.monotonic()-last>600:raise RuntimeError(f"no new render shards: {sorted(remaining)}")
            time.sleep(2)
        else:last=time.monotonic()
    print("all 30 house shards ready")

if __name__=="__main__":
    main()
