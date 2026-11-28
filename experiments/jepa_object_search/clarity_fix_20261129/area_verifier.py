"""Estimate how many current-view pixels belong to the remembered object, including its margin."""
from pathlib import Path
from functools import lru_cache
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REFINED = HERE.parent / 'refined_confirmation_20261129'
sys.path.insert(0, str(REFINED))
from memory_refit import RefinedMemory
from clarity_refit import crop_features, Trees

def pixel_features(rgb, depth, box, ref, pose):
    depth = np.asarray(depth, np.float32).squeeze()
    rgb = np.asarray(rgb)[...,:3].astype(np.float32,copy=False)
    l,t,r,b=map(int,box)
    w,h=r-l,b-t
    mx,my=max(2,round(.22*w)),max(2,round(.22*h))
    yy,xx=np.mgrid[max(0,t-my):min(192,b+my),max(0,l-mx):min(192,r+mx)]
    area=xx.size
    n=min(768,area)
    ids=((np.arange(n)+.5)*area/n).astype(int)
    yy,xx=yy.ravel()[ids],xx.ravel()[ids]
    z=depth[yy,xx]; color=rgb[yy,xx]
    qx=(xx+.5-l)/w; qy=(yy+.5-t)/h
    interior=(qx>=0)&(qx<1)&(qy>=0)&(qy<1)
    center=(abs(qx-.5)<.31)&(abs(qy-.5)<.31)
    if not center.any():center=interior
    kx, kz, angle=pose
    right=(xx+.5-96)*z/96
    up=(96-yy-.5)*z/96
    src_z=kz+right*np.sin(angle)+z*np.cos(angle)
    src_x=kx+right*np.cos(angle)-z*np.sin(angle)
    u0=np.floor(96+96*src_x/np.maximum(.01,src_z)).astype(int)
    v0=np.floor(96-96*up/np.maximum(.01,src_z)).astype(int)
    left,top,end,bottom=ref.sourcebox
    rsw=end-left
    if not hasattr(ref,'_area_index'):
        # Reanchoring can rebuild a sparse verifier with a different stride.
        # Recover each exact pixel location from its 3D camera point.
        zs=ref.xyz[:,2]
        src_u=np.floor(96+96*ref.xyz[:,0]/np.maximum(.001,zs)).astype(int)
        src_v=np.floor(96-96*ref.xyz[:,1]/np.maximum(.001,zs)).astype(int)
        grid=np.full((bottom-top,rsw),-1,int)
        valid=(src_u>=left)&(src_u<end)&(src_v>=top)&(src_v<bottom)&(zs>.05)
        grid[src_v[valid]-top,src_u[valid]-left]=np.flatnonzero(valid)
        ref._area_index=grid
    score=np.zeros(n); near=np.zeros(n); best=np.full(n,99.); best_w=np.zeros(n)
    for du,dv in ((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)):
        u=u0+du; v=v0+dv
        inside=(u>=left)&(u<end)&(v>=top)&(v<bottom)&(src_z>.05)
        idx=ref._area_index[(v-top).clip(0,bottom-top-1),(u-left).clip(0,rsw-1)]
        inside&=idx>=0
        dz=np.abs(src_z-ref.xyz[idx,2])/(.055+.035*np.maximum(.01,src_z))
        dc=np.abs(color-ref.rgb[idx]).mean(axis=1)/45.
        err=np.where(inside,dz+dc,99.)
        near=np.maximum(near,np.where(inside, np.exp(-np.minimum(15,dz)),0.))
        score=np.maximum(score,ref.w[idx]*np.exp(-np.minimum(15,err)))
        better=err<best
        best_w=np.where(better,ref.w[idx],best_w);best=np.minimum(best,err)
    center_z=np.median(z[center]) if center.any() else 0.
    center_c=np.median(color[center],axis=0) if center.any() else np.zeros(3)
    xborder=np.minimum(qx,1-qx);yborder=np.minimum(qy,1-qy)
    inside_src=np.minimum.reduce(((u0+.5-left)/rsw,(end-u0-.5)/rsw,
                                 (v0+.5-top)/(bottom-top),(bottom-v0-.5)/(bottom-top)))
    x=np.column_stack([score,near,best_w,np.minimum(12,best),xborder,yborder,
        inside_src, qx,qy,z,z-center_z, np.abs(color-center_c).mean(axis=1),
        np.abs(depth[yy,(xx-1).clip(0,191)]-z),np.abs(depth[(yy-1).clip(0,191),xx]-z),
        *[np.full(n,k) for k in (w,h,center_z)]])
    return np.nan_to_num(x.astype('float32')), yy,xx,interior,float(area)/n

def frame_features(pixel_scores,pixel_x,inside,scale,orig):
    p=pixel_scores; in_area=p[inside].sum()*scale; out_area=p[~inside].sum()*scale
    totals=[p.sum()*scale,in_area,out_area,(p>.5).sum()*scale,(p>.8).sum()*scale,
            (p*(1-p)).sum()*scale,*np.quantile(p,[.1,.25,.5,.75,.9])]
    for j in (0,1,2):
        totals.extend(np.mean(pixel_x[inside,j]>v) if inside.any() else 0. for v in (.25,.5,.75))
    return np.r_[totals,orig].astype('float32')

class AreaEstimate:
    @classmethod
    @lru_cache(None)
    def load(cls):
        self=cls()
        self.pixel=Trees(HERE/'private'/'fg.npz')
        self.area=Trees(HERE/'private'/'footprint.npz')
        self.binary=Trees(HERE/'private'/'clear.npz')
        self.old=Trees(REFINED/'private_clarity_model'/'clear_frame.npz')
        self.cfg=json.loads((HERE/'private'/'cal.json').read_text())
        return self

    def predict(self,rgb,depth,box,ref,pose,old):
        px,_,_,inside,scale=pixel_features(rgb,depth,box,ref,pose)
        px=px.astype('float16').astype('float32')
        orig=crop_features(rgb,depth,box,ref,pose,old)
        fv=frame_features(self.pixel(px),px,inside,scale,orig)[None]
        p=float(self.area(fv)[0])
        logit=np.log(np.clip(p,1e-7,1)/np.clip(1-p,1e-7,1))
        pa=1/(1+np.exp(-np.clip(self.cfg['a']*logit+self.cfg['b'],-32,32)))
        w=self.cfg['area_weight']
        current=w*pa+(1-w)*self.binary(fv)[0]
        return float(combine(current,self.old(orig[None])[0],self.cfg['old_weight'],self.cfg['fusion']))

def combine(current, prior, prior_weight, fusion):
    if fusion=='mean':
        return (1-prior_weight)*current+prior_weight*prior
    def logit(p):
        return np.log(np.clip(p,1e-7,1)/np.clip(1-p,1e-7,1))
    # Join consistent independent geometric and historical evidence, while still
    # allowing disagreement about a foreground region to reduce confidence.
    vote=float(fusion)*((1-prior_weight)*logit(current)+prior_weight*logit(prior))
    return 1/(1+np.exp(-np.clip(vote,-32,32)))

class AreaMemory(RefinedMemory):
    def __init__(self,source_rgb,source_depth,source_box):
        super().__init__(source_rgb,source_depth,source_box)
        self.clarity=AreaEstimate.load()
