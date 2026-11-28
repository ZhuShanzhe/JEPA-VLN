"""Online same-instance frame confirmation, using only a previous observed box, RGB-D and odometry."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent
MODELS=HERE/"private_confirm_models"

class Trees:
    def __init__(self, path):
        with np.load(path) as d:self.model={k:d[k] for k in d.files}
    def __call__(self,x):
        m=self.model
        n=len(x); nt=len(m["key"])
        nodes=np.zeros((n,nt),dtype=int)
        tr=np.arange(nt)[None,:]
        for _ in range(int(m["depth"])):
            key=m["key"][tr,nodes]
            left=m["left"][tr,nodes];right=m["right"][tr,nodes]
            small=x[np.arange(n)[:,None],np.maximum(key,0)]<=m["threshold"][tr,nodes]
            nodes=np.where(key<0,nodes,np.where(small,left,right))
        logit=m["initial"]+m["rate"]*m["value"][tr,nodes].sum(axis=1)
        return 1/(1+np.exp(-np.clip(logit,-40,40)))

@lru_cache(3)
def load(name):return Trees(MODELS/(name+".npz"))

@lru_cache(1)
def thresholds():
    return json.loads((MODELS/"thresholds.json").read_text())

def source_features(rgb,depth,box):
    rgb=np.asarray(rgb)[...,:3].astype(np.float32)
    depth=np.nan_to_num(depth,nan=0,posinf=0).astype(np.float32)
    l,t,r,b=map(int,box)
    if depth.shape!=(192,192) or not (0<=l<r<=192 and 0<=t<b<=192):raise ValueError("invalid native observation")
    yy,xx=np.mgrid[t:b,l:r]
    h,w=b-t,r-l
    z=depth[yy,xx]
    u=(xx+.5-l)/w; v=(yy+.5-t)/h
    mid=(abs(u-.5)<.26)&(abs(v-.5)<.26)&(z>.05)
    good=z>.05
    sample=z[mid] if mid.any() else z[good] if good.any() else np.array([0.])
    q=np.quantile(sample,[.15,.5,.85]);center=np.median(rgb[yy[mid],xx[mid]],axis=0) if mid.any() else np.zeros(3)
    border=np.minimum.reduce((u,1-u,v,1-v))
    ring=depth[max(0,t-2):min(192,b+2),max(0,l-2):min(192,r+2)]
    rg=ring[ring>.05]
    near=np.quantile(rg,.15) if rg.size else 0.
    colors=rgb[yy,xx]
    jump=np.maximum(abs(z-depth[yy,(xx-1).clip(0,191)]),abs(z-depth[(yy-1).clip(0,191),xx]))
    ext=[np.full(z.shape,val) for val in (w/192,h/192,w/max(h,1),float(np.median(sample)))]
    x=np.column_stack([a.ravel() for a in (u,v,abs(u-.5),abs(v-.5),border,z,z-q[0],z-q[1],z-q[2],z-near,
        np.mean(abs(colors-center),axis=2),jump,np.minimum(w,h)*border,*ext)])
    return x,xx.ravel(),yy.ravel()

def empty():
    return dict(accept=False,candidate=False,box=None,estimated_pixels=0.,p_clear=0.,p_same=0.,reason="no_supported_surface")

class FrameConfirmer:
    def __init__(self,source_rgb,source_depth,source_box):
        src=np.asarray(source_rgb)[...,:3]
        dz=np.asarray(source_depth).squeeze()
        x,xx,yy=source_features(src,dz,source_box)
        # Deterministic subsampling bounds per-frame cost for very large cues.
        idx=np.arange(0,len(x),max(1,int(np.ceil(len(x)/6000))))
        self.area_weight=len(x)/len(idx)
        self.w=load("source")(x[idx]);self.w[dz[yy[idx],xx[idx]]<=.05]=0
        x0,y0=xx[idx],yy[idx]
        z=dz[y0,x0].astype(np.float32)
        self.xyz=np.column_stack(((x0+.5-96)*z/96,(96-y0-.5)*z/96,z))
        self.rgb=src[y0,x0].astype(np.float32)
        self.sourcebox=tuple(map(int,source_box))
        self.start_area=float(self.w.sum()*self.area_weight)

    def evidence(self,rgb,depth,pose):
        rgb=np.asarray(rgb)[...,:3].astype(np.float32)
        depth=np.asarray(depth).squeeze()
        x,z,yaw=map(float,pose)
        px,py,pz=self.xyz.T
        dx,dz=px-x,pz-z
        sn,cs=np.sin(yaw),np.cos(yaw)
        fw=-dx*sn+dz*cs;right=dx*cs+dz*sn
        uf=96+96*right/np.maximum(fw,.01)
        vf=96-96*py/np.maximum(fw,.01)
        ok=(fw>.12)&(uf>=0)&(uf<192)&(vf>=0)&(vf<192)&(self.w>.12)
        if not ok.any():return None,None
        ui=uf[ok].astype(int);vi=vf[ok].astype(int)
        projected=fw[ok];source_z=pz[ok];w=self.w[ok];colors=self.rgb[ok]
        # Find a matching observed surface within one pixel of its predicted projection.
        def residual(offx,offy):
            u=(ui+offx).clip(0,191);v=(vi+offy).clip(0,191)
            dd=abs(depth[v,u]-projected)
            cd=np.mean(abs(colors-rgb[v,u]),axis=1)
            return dd,cd
        depth0,col0=residual(0,0)
        best=depth0/(.045+.025*projected)+col0/42
        dep=depth0.copy();color=col0.copy()
        for off in ((0,1),(1,0),(-1,0),(0,-1)):
            d,c=residual(*off);loss=d/(.045+.025*projected)+c/42
            improve=loss<best;best=np.minimum(best,loss)
            dep=np.where(improve,d,dep);color=np.where(improve,c,color)
        surface=(dep<.07+.045*projected)&(color<68)
        selected=surface&(w>=.35)
        if selected.sum()<5:return None,None
        # A conservative estimated bounding rectangle includes almost all supported samples.
        u,v=ui[selected],vi[selected]
        q=.008 if len(u)>=120 else 0.
        l,r=np.quantile(u,[q,1-q]);t,b=np.quantile(v,[q,1-q]);r+=1;b+=1
        box=(int(l),int(t),min(192,int(np.ceil(r))),min(192,int(np.ceil(b))))
        width=box[2]-box[0];height=box[3]-box[1];box_area=width*height
        expansion=(source_z/projected).clip(.15,6)**2
        area=float((w*surface*expansion).sum()*self.area_weight)
        base=max(.01,self.w.sum())
        f=[area,box_area,self.start_area,min(box_area,area),width,height,area/max(1,box_area),
           w.sum()/base,w[surface].sum()/base,np.mean(surface),np.mean(selected),
           np.mean(col0[selected]),np.mean(dep[selected]),np.mean(color[selected]),
           np.mean(projected[selected]),np.std(projected[selected]),np.median(expansion[selected]),
           abs(x),z,abs(yaw),float(l<2 or t<2 or r>190 or b>190),
           (self.sourcebox[2]-self.sourcebox[0])*(self.sourcebox[3]-self.sourcebox[1])]
        return np.nan_to_num(np.array(f,np.float32)),box

    def observe(self,rgb,depth,pose):
        f,box=self.evidence(rgb,depth,pose)
        if f is None:return empty()
        clear=float(load("clear")(f[None,:])[0]);same=float(load("same")(f[None,:])[0])
        accept=bool(clear*same>=thresholds()["threshold"])
        return dict(accept=accept,candidate=True,box=box,estimated_pixels=float(f[0]),p_clear=clear,p_same=same,
                    reason="verified_source_surface" if accept else "insufficient_identity_or_clarity")
