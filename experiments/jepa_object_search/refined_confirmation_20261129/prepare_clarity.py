"""Construct current-crop area examples, including missed clear views and cluttered references."""
from pathlib import Path
import argparse
import os
import numpy as np
from clarity_refit import HERE,PRIOR,crop_features
from frame_confirm import FrameConfirmer


def labels(mask,box):
    pixels=int(mask.sum())
    if not pixels:return 0,False
    yy,xx=np.where(mask);left,top,right,bottom=map(int,box)
    r,l,b,t=xx.max()+1,xx.min(),yy.max()+1,yy.min()
    intersection=max(0,min(r,right)-max(l,left))*max(0,min(b,bottom)-max(t,top))
    iou=intersection/max(1.,(r-l)*(b-t)+(right-left)*(bottom-top)-intersection)
    return pixels,mask[top:bottom,left:right].sum()>=pixels*.5 and iou>=.3


def one(file,dst):
    if dst.exists():return
    with np.load(file) as d:
        rgb,depth,mask,poses,box=(d[k] for k in ('rgb','depth','mask','pose','box'))
    verifier=FrameConfirmer(rgb[0],depth[0],box)
    masks=np.unpackbits(mask,axis=1).reshape(-1,192,192)
    feats=[];info=[]
    for i,pose in enumerate(poses):
        old=verifier.observe(rgb[i],depth[i],pose)
        if old['box'] is None:continue
        pix,spatial=labels(masks[i],old['box'])
        feats.append(crop_features(rgb[i],depth[i],old['box'],verifier,pose,old))
        info.append([i,pix,spatial,old['p_clear'],old['p_same'],old['estimated_pixels']])
    dst.parent.mkdir(exist_ok=True,parents=True)
    temp=dst.with_suffix('.partial.npz')
    np.savez_compressed(temp,x=feats,labels=info,source_pixels=masks[0].sum(),
                        source_box_area=(box[2]-box[0])*(box[3]-box[1]),frames=len(poses))
    temp.replace(dst)


def all_files():
    rows=[]
    for split in ('train','val'):
        root=PRIOR/'private_confirm_data'/split
        for src in sorted(root.glob('*/*.npz')):
            for label,folder in [('recorded',root),('oblique',HERE/'private_clarity_views'/split)]:
                rel=src.relative_to(root)
                rows.append((folder/rel,HERE/'private_clarity_features'/label/split/rel))
    assert len(rows)==864
    return rows


if __name__=='__main__':
    os.umask(0o077)
    p=argparse.ArgumentParser();p.add_argument('index',type=int);p.add_argument('workers',type=int);a=p.parse_args()
    for src,dst in all_files()[a.index::a.workers]:
        one(src,dst)
