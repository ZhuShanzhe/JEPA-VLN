"""Recover current to first-view correspondences on the existing training and calibration sweeps."""
from pathlib import Path
import os
import sys
import numpy as np
from area_verifier import HERE,pixel_features
from prepare_clarity import all_files
from frame_confirm import FrameConfirmer

def process(index,source,feature):
    split=feature.parent.parent.name
    dest=HERE/'private'/'features'/split/feature.parent.name/(str(index)+'.npz')
    if dest.exists():return
    with np.load(feature) as d: originals,y=d['x'],d['labels']
    if not len(y):return
    with np.load(source) as d:
        rgbs,depths,poses,flags,box=[d[k] for k in ('rgb','depth','pose','mask','box')]
    masks=np.unpackbits(flags,axis=1).reshape(-1,192,192)
    ref=FrameConfirmer(rgbs[0],depths[0],box)
    arr=[];gt=[];interior=[];offset=[0];scales=[]
    for i in y[:,0].astype(int):
        e=ref.observe(rgbs[i],depths[i],poses[i])
        x,v,u,inside,scale=pixel_features(rgbs[i],depths[i],e['box'],ref,poses[i])
        arr.append(x);gt.append(masks[i,v,u]);interior.append(inside);scales.append(scale);offset.append(offset[-1]+len(x))
    dest.parent.mkdir(exist_ok=True,parents=True)
    partial=dest.with_suffix('.partial.npz')
    np.savez(partial,px=np.concatenate(arr).astype('float16'),gt=np.concatenate(gt),interior=np.concatenate(interior),offset=offset,
             scale=scales,orig=originals,labels=y)
    partial.replace(dest)

if __name__=='__main__':
    os.umask(0o077)
    rank,n=map(int,sys.argv[1:])
    for idx,(source,dest) in list(enumerate(all_files()))[rank::n]:
        process(idx,source,dest)
