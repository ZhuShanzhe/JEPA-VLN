"""Encode true first-discovery instance masks in the frozen V11 visual subspace."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def grid_overlaps(pixel_count=128, cell_count=7):
    """Exact overlap length between a pixel and each feature cell on one axis."""
    x=np.arange(pixel_count)[None,:]
    lo=np.arange(cell_count)[:,None]*pixel_count/cell_count
    return np.maximum(0,np.minimum(x+1,lo+pixel_count/cell_count)-np.maximum(x,lo)).astype("float32")


def encode_file(enc,areas,masks,raw,dest):
    with np.load(masks) as z:
        fields={key:z[key] for key in z.files if key!="pack_mask"}
        evidence=np.unpackbits(z["pack_mask"],axis=-1).astype("float32")
    taps=fields["first_tap"]; valid=taps>=0
    with np.load(raw) as z:
        future=z["future"]
        assert np.array_equal(z["words"],fields["words"]) and np.array_equal(z["y"],fields["y"])
    # Future images are teacher-side. They never enter the actor data path.
    keys=sorted({(r,int(taps[r,q])) for r,q in zip(*np.where(valid))})
    frames=[future[r,t] for r,t in keys]
    enc.encode(frames,enc.words(fields["words"].tolist()))
    dense={k:enc.cache[hashlib.sha256(img.tobytes()).digest()][1] for k,img in zip(keys,frames)}
    weights=(areas @ evidence @ areas.T).reshape(*valid.shape,49)
    pixel_count=evidence.sum((-1,-2)).astype("int32")
    assert np.allclose(weights.sum(-1),pixel_count,rtol=1e-5,atol=0.01)
    assert np.array_equal(pixel_count>=32,valid)
    proj=enc.proj.float().cpu().numpy()
    output=np.zeros((*valid.shape,proj.shape[1]),np.float32)
    for r,q in zip(*np.where(valid)):
        v=(weights[r,q]/pixel_count[r,q]) @ dense[r,int(taps[r,q])].astype("float32")
        v=v @ proj
        norm=float(np.linalg.norm(v))
        if not np.isfinite(norm) or norm<=1e-8:
            raise ValueError("empty feature for visible target")
        output[r,q]=v/norm
    assert np.isfinite(output).all() and np.allclose(np.linalg.norm(output[valid],axis=-1),1.,atol=1e-5)
    part=dest.with_suffix(".pending.npz")
    np.savez_compressed(part,**fields,pixels=pixel_count,local_z=output.astype("float16"))
    part.replace(dest)
    if len(enc.cache)>450:
        enc.cache.clear()


def summarize(folder):
    houses={}
    classes=Counter()
    pixel=[]
    for path in sorted(folder.glob("*.npz")):
        with np.load(path) as z:
            h=str(z["house"])
            count=houses.setdefault(h,Counter())
            y,taps,pairs=z["y"],z["first_tap"],z["pair_queries"]
            assert np.array_equal(y,np.count_nonzero(z["hit_ids"],axis=-1))
            words=z["words"].tolist()
            assert len(set(words))==len(words)
            for a,b in pairs:
                assert not np.any((y[:,a]>0)&(y[:,b]>0))
            count.update(origins=1,queries=len(words),visible_route_queries=int((y>0).sum()),
                         zero_routes=int((y==0).sum()),distinct_instances_summed_over_route_queries=int(y.sum()),
                         disjoint_pairs=len(pairs),junctions_with_disjoint_pairs=int(bool(len(pairs))))
            classes.update(words);pixel.extend(z["pixels"][taps>=0].tolist())
    total=Counter()
    for v in houses.values():total.update(v)
    return dict(counts=dict(total),houses=houses,n_houses=len(houses),target_query_counts=classes,
                first_instance_pixels_quartiles=np.quantile(pixel,[0,.25,.5,.75,1]).tolist())


def main(root):
    sys.path.insert(0,str(root/"jepa_hidden_v11/models"))
    from encode import Encoder
    torch.set_num_threads(2)
    enc=Encoder(root);areas=grid_overlaps()
    raw=root/"jepa_hidden_v11/hssd"
    stage=root/"jepa_route_v12/data"
    cohort=json.loads((raw/"cohort.json").read_text())
    for split in ("train","calibration"):
        dest=stage/"local_targets"/split
        dest.mkdir(parents=True,exist_ok=True)
        expected=sum(json.loads((raw/split/h/"receipt.json").read_text())["origins"] for h in cohort[split])
        while True:
            masks=stage/"region_masks"/split
            for f in sorted(masks.glob("*/*.npz")):
                if f.name.endswith(".partial.npz") or (dest/f.name).exists():
                    continue
                encode_file(enc,areas,f,raw/split/f.parent.name/f.name,dest/f.name)
            if len(list(dest.glob("*.npz")))==expected and all((masks/h/"done.json").exists() for h in cohort[split]):
                break
            time.sleep(3)
        stats=summarize(dest)
        (stage/("stats_"+split+".json")).write_text(json.dumps(stats,indent=2))
        print(split, stats["counts"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("project",type=Path)
    main(p.parse_args().project)
