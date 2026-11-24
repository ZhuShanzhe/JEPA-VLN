"""Encode box-centered first-discovery crops and check true-goal text margins."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def crop_bounds(mask):
    y,x=np.where(mask)
    height,width=mask.shape
    # Keep the complete visible object. A square crop avoids stretching its shape.
    side=min(max(height,width),max(16,int(np.ceil(1.25*max(np.ptp(x)+1,np.ptp(y)+1)))))
    x0=int(np.clip(round((x.max()+x.min()+1-side)/2),0,width-side))
    y0=int(np.clip(round((y.max()+y.min()+1-side)/2),0,height-side))
    return y0,x0,y0+side,x0+side


def main(root):
    path=root/"jepa_hidden_v11"
    stage=root/"jepa_route_v12/data"
    sys.path.insert(0,str(path/"models"))
    from encode import Encoder
    device=Encoder(root)
    torch.set_num_threads(2)
    result={}
    for split in ("train","calibration"):
        sampled=defaultdict(list)
        dest=stage/"crop_targets"/split
        dest.mkdir(exist_ok=True,parents=True)
        for f in sorted((stage/"region_masks"/split).glob("*/*.npz")):
            with np.load(f) as z:
                mask=np.unpackbits(z["pack_mask"],axis=-1)
            with np.load(stage/"local_targets"/split/f.name) as z:
                pooled=z["local_z"]
                words,taps,y=z["words"],z["first_tap"],z["y"]
            parent_raw=path/"hssd"/split/f.parent.name/f.name
            key=hashlib.sha256(str(parent_raw.relative_to(path)).encode()).hexdigest()
            with np.load(parent_raw) as z: future=z["future"]
            cache=path/"models/cache/hssd"/split
            with np.load(cache/"actor"/(key+".npz")) as z:
                assert np.array_equal(z["words"],words)
                text=z["text"].astype("float32")
            with np.load(cache/"teacher"/(key+".npz")) as z:
                global_features=z["future"][:,:,0].astype("float32")
            images=[];indices=[];sides=[]
            for route,q in zip(*np.where(taps>=0)):
                ya,xa,yb,xb=crop_bounds(mask[route,q])
                images.append(device.clip.prep(Image.fromarray(future[route,taps[route,q],ya:yb,xa:xb])))
                sides.append(yb-ya);indices.append((route,q))
            with torch.inference_mode():
                v=device.net.visual(torch.stack(images).to(device.device)).float()
                vector=F.normalize(v@device.proj,dim=-1).cpu().numpy()
            vectors=np.zeros_like(pooled)
            for at,(route,q) in enumerate(indices):
                vectors[route,q]=vector[at]
                # Eligible alternatives were hidden at the start and genuinely
                # absent from this route's four evaluated views.
                alternatives=np.flatnonzero(y[route]==0)
                if len(alternatives):
                    for label,value in (("crop",vector[at]),("mask_grid",pooled[route,q]),("whole",global_features[route,taps[route,q]])):
                        logits=text@value.astype("float32")
                        sampled[label].append(float(np.mean(logits[q]>logits[alternatives])))
                sampled["box_side_pixels"].append(sides[at])
            destfile=dest/f.name
            part=destfile.with_suffix(".partial.npz")
            np.savez_compressed(part,words=words,first_tap=taps,local_z=vectors)
            part.replace(destfile)
        result[split]={name:dict(n=len(vals),mean=float(np.mean(vals)),median=float(np.median(vals))) for name,vals in sampled.items()}
        print(split,result[split],flush=True)
    (stage/"crop_quality.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("project",type=Path)
    main(p.parse_args().project)
