"""Native level RGB-D camera, hfov 90 degrees, height 1.31m. Path features in its source frame."""
from functools import lru_cache
import math
import numpy as np

NAMES=("right_m","forward_m","sin_yaw","cos_yaw","is_forward","path_coverage",
       "left_coverage","right_coverage","occluded_fraction","left_occluded","right_occluded",
       "depth_margin_p10","depth_margin_median","body_hit","left_hit","right_hit","observed_clear")
SIDE=160
CELL=.10


def from_depth(depth):
    depth=np.asarray(depth,dtype=np.float32).squeeze()
    if depth.ndim!=2 or depth.shape[0]!=depth.shape[1]:
        raise ValueError("Expected a square forward depth image")
    h,w=depth.shape
    u,v=np.meshgrid(np.arange(w)+.5,np.arange(h)+.5)
    x=(u-w/2)*depth/(w/2)
    up=(h/2-v)*depth/(w/2)+1.31
    valid=np.isfinite(depth)&(depth>.05)&(depth<SIDE*CELL)&(abs(x)<SIDE*CELL)&(up>=.10)&(up<=1.40)
    occupancy=np.zeros((SIDE,SIDE*2),bool)
    occupancy[np.floor(depth[valid]/CELL).astype(int),np.floor(x[valid]/CELL).astype(int)+SIDE]=True
    return depth,occupancy


@lru_cache(2048)
def nominal(plan):
    x=z=yaw=0.
    out=[]
    for a in plan:
        if a==1:
            x-=.25*math.sin(yaw)
            z+=.25*math.cos(yaw)
        elif a in (2,3):
            yaw+=(1 if a==2 else -1)*math.pi/6
        else:raise ValueError("Unexpected primitive")
        out.append((x,z,math.sin(yaw),math.cos(yaw),float(a==1)))
    return np.array(out)


def path_features(plan,source_depth):
    """Return [actual steps,17], no padding. Accept either depth or output of from_depth."""
    src=source_depth if isinstance(source_depth,tuple) else from_depth(source_depth)
    depth,grid=src
    n=len(plan); sz=depth.shape[0]
    base=nominal(tuple(map(int,plan)))
    x,z,sin,cos,move=base.T
    # Front and rear halves of each traversed 25cm segment, two flanks, four body heights.
    offset,side,height=np.meshgrid([-.20,-.08,.04],[-.17,-.08,.08,.17],[-1.13,-.85,-.43,0.],indexing="ij")
    off,side,ht=offset.ravel()[None,:],side.ravel()[None,:],height.ravel()[None,:]
    px=x[:,None]-off*sin[:,None]+side*cos[:,None]
    pz=z[:,None]+off*cos[:,None]+side*sin[:,None]
    uu=(sz/2+sz/2*px/np.maximum(.001,pz))
    vv=(sz/2-sz/2*ht/np.maximum(.001,pz))
    inside=(pz>.05)&(uu>=0)&(vv>=0)&(uu<sz)&(vv<sz)
    measured=depth[np.clip(vv.astype(int),0,sz-1),np.clip(uu.astype(int),0,sz-1)]
    observed=inside & np.isfinite(measured) & (measured>.05)
    delta=np.where(observed,measured-pz,0.)
    occluded=observed&(delta<-.12)
    gx=np.floor(px/CELL).astype(int)+SIDE; gz=np.floor(pz/CELL).astype(int)
    ingrid=(gx>=0)&(gx<SIDE*2)&(gz>=0)&(gz<SIDE)
    hits=ingrid & grid[np.clip(gz,0,SIDE-1),np.clip(gx,0,SIDE*2-1)]
    left=side.ravel()<0
    masked=np.where(observed,np.clip(delta,-4.,4.),np.inf)
    ordered=np.sort(masked,axis=1); count=observed.sum(axis=1)
    def margin(fraction):
        index=(fraction*np.maximum(count-1,0)).astype(int)
        return np.where(count>0,ordered[np.arange(n),index],0.)
    def occ(m,mask):
        return m[:,mask].sum(1)/np.maximum(1,observed[:,mask].sum(1))
    result=np.column_stack((base,observed.mean(1),observed[:,left].mean(1),observed[:,~left].mean(1),
                            occ(occluded,np.ones(len(left),bool)),occ(occluded,left),occ(occluded,~left),
                            margin(.10),margin(.50),hits.any(1),hits[:,left].any(1),hits[:,~left].any(1),
                            (observed&~occluded).mean(1)))
    assert result.shape==(n,len(NAMES)) and np.isfinite(result).all()
    return result.astype(np.float32)
