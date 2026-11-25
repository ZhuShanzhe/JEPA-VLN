"""Advisory different-instance evidence from disjoint reprojection at nearby cameras."""
import itertools
import math
import numpy as np


def possible_box(src,dst,idx):
    """Project an enlarged set of source rays at a range of plausible surface depths."""
    box=np.asarray(src["boxes"][idx],dtype=float)
    K,other=np.asarray(src["K"]),np.asarray(dst["K"])
    T,U=np.asarray(src["c2w"]),np.asarray(dst["c2w"])
    point=src["points"][idx]
    if point is None or not np.isfinite(point).all():
        return None
    z=(T[:3,:3].T@(point-T[:3,3]))[2]
    w,h=box[2:]-box[:2]
    width,height=src["image_size"]
    if min(w,h)<10 or min(box[0],box[1],width-box[2],height-box[3])<3 or z<.4:
        return None
    # Emit no exclusion for very large objects filling the frame or severe truncation.
    if w>width*.60 or h>height*.60:
        return None
    dz=max(.15,.08*z,.65*z*max(w/K[0,0],h/K[1,1]))
    left,top=box[:2]-.25*np.array([w,h])-4
    right,bottom=box[2:]+.25*np.array([w,h])+4
    xy=[]
    for u,v,d in itertools.product((left,right),(top,bottom),(max(.15,z-dz),z+dz)):
        p=T[:3,:3]@(np.linalg.solve(K,[u,v,1.])*d)+T[:3,3]
        q=U[:3,:3].T@(p-U[:3,3])
        if q[2]<=.1:
            return None
        xy.append((other@q)[:2]/q[2])
    pts=np.array(xy)
    return np.r_[pts.min(0),pts.max(0)]


def soft_separation(left,right):
    """This signal suggests an independent check. It cannot prove different identity."""
    L,R=np.asarray(left["c2w"]),np.asarray(right["c2w"])
    distance=np.linalg.norm(L[:3,3]-R[:3,3])
    degrees=math.degrees(math.acos(np.clip(L[:3,2]@R[:3,2],-1.,1.)))
    if left["frame_id"]==right["frame_id"] or distance>.65 or degrees>30 or (distance<.03 and degrees<1.):
        return set()
    def gap(a,b):
        return np.max(np.maximum(a[:2]-b[2:],b[:2]-a[2:]))
    first=[possible_box(left,right,i) for i in range(len(left["boxes"]))]
    second=[possible_box(right,left,j) for j in range(len(right["boxes"]))]
    return {(i,j) for i,p in enumerate(first) if p is not None
            for j,q in enumerate(second) if q is not None and
            gap(p,np.array(right["boxes"][j]))>8 and gap(q,np.array(left["boxes"][i]))>8}
