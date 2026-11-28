"""Freeze the current RGB and observed cue crop; encode future RGB separately for supervision only."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"pose_horizon_20261128"))
from long_data import FrozenImageEncoder,cohorts


def encode(folder,out,encoder):
    with np.load(folder/"controls.npz",allow_pickle=False) as z:
        rows={k:z[k] for k in z.files}
    n=len(rows["reach"])
    sources=rows["source_index"]
    src=[f"{int(i)}_source.jpg" for i in sources]
    crop=[f"{int(i)}_crop.jpg" for i in sources]
    names=sorted((set(src)|set(crop)|set(rows["future_file"]))-{""})
    digests={name:hashlib.sha256((folder/name).read_bytes()).digest() for name in names}
    unique={v:k for k,v in digests.items()}
    keys=list(unique)
    features={}
    for start in range(0,len(keys),64):
        subset=keys[start:start+64]
        ims=[]
        for k in subset:
            with Image.open(folder/unique[k]) as im:
                ims.append(np.asarray(im.convert("RGB")))
        features.update(zip(subset,encoder.encode(ims)))
    def f(name):
        return features[digests[name]] if name else np.zeros(1024,np.float16)
    context=np.stack([np.concatenate((f(s),f(c))) for s,c in zip(src,crop)]).astype(np.float16)
    future=np.stack([f(k) for k in rows["future_file"]]).astype(np.float16)
    rows.update(context=context,future=future,future_valid=rows["reach"].astype(bool),
                pose=rows["pose"].astype(np.float32))
    temp=out.with_suffix(".partial.npz")
    np.savez_compressed(temp,**rows);temp.replace(out)
    print("encoded",folder.name,n,flush=True)


if __name__=="__main__":
    torch.set_num_threads(2)
    os.umask(0o077)
    dst=HERE/"private_data";dst.mkdir(exist_ok=True)
    pending={split+"_"+h for _,split,houses in cohorts() for h in houses}
    model=FrozenImageEncoder(HERE.parent)
    last=time.monotonic()
    while pending:
        ready=sorted(name for name in pending if (HERE/"private_render"/name/"controls.npz").exists())
        for name in ready:
            out=dst/(name+".npz")
            if not out.exists():encode(HERE/"private_render"/name,out,model)
            pending.remove(name)
        if ready:last=time.monotonic()
        if time.monotonic()-last>600:raise RuntimeError("No incoming house completion for ten minutes")
        if not ready:time.sleep(1)
    print("all 30 houses encoded",flush=True)
