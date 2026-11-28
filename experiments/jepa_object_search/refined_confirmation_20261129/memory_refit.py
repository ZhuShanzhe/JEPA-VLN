"""Trust an initial surface memory or acquire another physically displaced view before promotion."""
from functools import lru_cache
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'reactive_view_20261129'
sys.path.insert(0,str(OLD))
from frame_confirm import FrameConfirmer, Trees, thresholds, empty
MODELS=HERE/'private_memory_models'

@lru_cache(None)
def model(name):
    return Trees(MODELS/(name+'.npz'))

@lru_cache(None)
def gates():
    return json.loads((MODELS/'gates.json').read_text())

def dense(rgb,depth,box):
    """RGB-D contrast to the outer box and four border strips measured at the current view."""
    rgb=np.asarray(rgb,dtype=np.float32)[...,:3]; dep=np.asarray(depth,dtype=np.float32).squeeze()
    l,t,r,b=map(int,box);w=r-l;h=b-t
    y,x=np.mgrid[t:b,l:r]
    stride=max(1,int(np.ceil(x.size/4500)))
    y=y.ravel()[::stride];x=x.ravel()[::stride]; d=dep[y,x]; c=rgb[y,x]
    u=(x+.5-l)/w;v=(y+.5-t)/h
    f=[u,v,np.minimum(u,1-u),np.minimum(v,1-v), np.minimum.reduce((u,1-u,v,1-v)),d]
    for a,z,br,bt in [(max(0,l-max(2,w//5)),t,l,b),(r,t,min(192,r+max(2,w//5)),b),(l,max(0,t-max(2,h//5)),r,t),(l,b,r,min(192,b+max(2,h//5))),(l+w//3,t+h//3,max(l+w//3+1,r-w//3),max(t+h//3+1,b-h//3))]:
        dc=dep[z:bt,a:br].ravel();cc=rgb[z:bt,a:br].reshape(-1,3)
        valid=np.isfinite(dc)&(dc>.05)
        if valid.any():
            med=np.median(dc[valid]); color=np.median(cc[valid],axis=0)
            f.extend((d-med,np.abs(c-color).mean(axis=1)))
        else:f.extend((np.full_like(d,9),np.full_like(d,255)))
    near=np.quantile(d,[.1,.5,.9])
    f+=[d-val for val in near]
    for x2,y2 in [(x-1,y),(x,y-1)]:
        x2=x2.clip(0,191);y2=y2.clip(0,191)
        f.extend([abs(d-dep[y2,x2]),np.abs(c-rgb[y2,x2]).mean(axis=1)])
    f+=[np.full_like(d,a) for a in (w,h,np.mean(u<.1),np.ptp(near))]
    return np.nan_to_num(np.stack(f,axis=1),nan=0,posinf=30,neginf=-30),x,y

class Surface:
    def __init__(self,rgb,depth,box):
        self.rgb=np.asarray(rgb)[...,:3];self.depth=np.asarray(depth).squeeze()
        self.box=tuple(map(int,box));f,self.x,self.y=dense(rgb,depth,box)
        self.p=model('foreground')(f)
        z=self.depth[self.y,self.x]
        self.p[~np.isfinite(z)|(z<=.05)]=0.
        self.xyz=np.column_stack(((self.x+.5-96)*z/96,(96-self.y-.5)*z/96,z))
        self.colors=self.rgb[self.y,self.x].astype(np.float32)
        self.area=(self.box[2]-self.box[0])*(self.box[3]-self.box[1])
    def summary(self):
        p=self.p;z=self.xyz[:,2]
        return [self.area,self.box[2]-self.box[0],self.box[3]-self.box[1],p.mean(),(p>.6).mean(),(p>.8).mean(),
                self.area*p.mean(),*np.quantile(p,[.15,.5,.85]),*np.quantile(z,[.1,.5,.9]),np.std(self.colors,axis=0).mean()]
    def reproject(self,rgb,depth,pose):
        """Project the whole source box. Association evidence comes from agreement of independent foreground estimates."""
        x,z,a=pose
        dx=self.xyz[:,0]-x;dz=self.xyz[:,2]-z
        fwd=-dx*np.sin(a)+dz*np.cos(a); right=dx*np.cos(a)+dz*np.sin(a)
        u=np.floor(96+96*right/np.maximum(.01,fwd)).astype(int)
        v=np.floor(96-96*self.xyz[:,1]/np.maximum(.01,fwd)).astype(int)
        ons=(fwd>.1)&(u>=0)&(u<192)&(v>=0)&(v<192)&(self.p>.25)
        if ons.sum()<8:return None
        ix=np.flatnonzero(ons);u=u[ix];v=v[ix];fw=fwd[ix]
        dd=abs(depth[v,u]-fw);color=np.abs(rgb[v,u,:3].astype(float)-self.colors[ix]).mean(axis=1)
        ok=(dd<.07+.035*fw)&(color<65)
        if ok.sum()<8:return None
        ids=ix[ok];u=u[ok];v=v[ok];a,b=np.quantile(u,[.01,.99]);c,d=np.quantile(v,[.01,.99])
        box=(max(0,int(a)-1),max(0,int(c)-1),min(192,int(np.ceil(b))+2),min(192,int(np.ceil(d))+2))
        if min(box[2]-box[0],box[3]-box[1])<3:return None
        other=Surface(rgb,depth,box)
        # Densify only for lookup. Rows skipped by bounded subsampling use the nearest sampled pixel.
        inner=(u>=box[0])&(u<box[2])&(v>=box[1])&(v<box[3]);ids=ids[inner]
        if len(ids)<6:return None
        u=u[inner];v=v[inner]
        index=(v-box[1])*(box[2]-box[0])+u-box[0]
        stride=max(1,int(np.ceil(other.area/4500)))
        p2=other.p[(index//stride).clip(0,len(other.p)-1)];p1=self.p[ids]
        agree=(p1>=.55)&(p2>=.55)
        feature=[*self.summary(),*other.summary(),float(np.hypot(x,z)),abs(pose[2]),np.mean(ok),
                 len(ids)/len(self.p),p1.mean(),p2.mean(),np.mean(agree),np.count_nonzero(agree),np.median(dd[ok]),np.median(color[ok]),
                 np.mean(self.colors[ids],axis=0).mean()]
        return np.asarray(feature,float),other
    def verifier(self):
        v=FrameConfirmer(self.rgb,self.depth,self.box)
        v.xyz=self.xyz;v.rgb=self.colors;v.w=self.p;v.area_weight=self.area/len(self.p);v.start_area=float(v.w.sum()*v.area_weight)
        return v

def risk_features(source,tracker):
    n=source.area
    stride=max(1,int(np.ceil(n/6000))); stride2=max(1,int(np.ceil(n/4500)))
    ids=(np.arange(0,n,stride)//stride2).clip(0,len(source.p)-1)
    p=source.p[ids];q=tracker.w
    foreground=q>=.35
    return [*source.summary(),q.mean(),foreground.mean(),np.mean(q-p),np.mean(abs(p-q)),
            (p[foreground].mean() if foreground.any() else 0.),
            np.mean(foreground&(p>=.65)),np.mean((~foreground)&(p>=.65)),tracker.start_area]

def corroborate(source,observed,evidence,rgb,depth,pose):
    """Independently resegment the projected candidate in the new RGB-D view, then compare foreground correspondences."""
    if observed is None:return None
    other=Surface(rgb,depth,observed)
    x,z,a=pose;dx=source.xyz[:,0]-x;dz=source.xyz[:,2]-z
    fw=-dx*np.sin(a)+dz*np.cos(a);right=dx*np.cos(a)+dz*np.sin(a)
    u=np.floor(96+96*right/np.maximum(.01,fw)).astype(int);v=np.floor(96-96*source.xyz[:,1]/np.maximum(.01,fw)).astype(int)
    l,t,r,b=map(int,observed)
    ok=(fw>.12)&(u>=l)&(u<r)&(v>=t)&(v<b)&(source.p>.25)
    if ok.sum()<5:return None
    ix=np.flatnonzero(ok);u=u[ix];v=v[ix]
    dst=other.p[(((v-t)*(r-l)+u-l)//max(1,int(np.ceil(other.area/4500)))).clip(0,len(other.p)-1)]
    dd=abs(depth[v,u]-fw[ix]);cd=abs(rgb[v,u,:3].astype(float)-source.colors[ix]).mean(axis=1)
    match=(dd<.07+.045*fw[ix])&(cd<65)
    agree=match&(dst>=.45)&(source.p[ix]>=.45)
    if agree.sum()<4:return None
    q=np.array([agree.sum(),len(ix),agree.mean(),match.mean(),np.mean(dst),np.mean(source.p[ix]),
              np.hypot(x,z),abs(a),np.mean(cd[agree]),np.median(dd[agree])])
    return np.r_[source.summary(),other.summary(),evidence,q],other

def relative(current,anchor):
    x,z,a=anchor;dx=current[0]-x;dz=current[1]-z
    b=current[2]-a
    return (dx*np.cos(a)+dz*np.sin(a),-dx*np.sin(a)+dz*np.cos(a),np.arctan2(np.sin(b),np.cos(b)))

class RefinedMemory:
    def __init__(self,source_rgb,source_depth,source_box,*,use_memory=True,use_clarity=True):
        self.use_memory=use_memory; self.use_clarity=use_clarity
        self.verifier=FrameConfirmer(source_rgb,source_depth,source_box)
        self.anchor=(0,0,0);self.state='trusted_initial';self.initial_confidence=1.;self.secondary=None
        if use_memory:
            self.source=Surface(source_rgb,source_depth,source_box)
            self.initial_confidence=float(model('initial')(np.asarray([self.source.summary()]))[0])
            self.contamination_risk=float(model('contamination')(np.asarray([risk_features(self.source,self.verifier)]))[0])
            if self.initial_confidence<gates()['initial'] and self.contamination_risk>=gates()['contamination']:
                self.state='pending'
        if use_clarity:
            from clarity_refit import ClarityEstimator
            self.clarity=ClarityEstimator.load()
    def observe(self,rgb,depth,pose):
        depth=np.asarray(depth,dtype=np.float32).squeeze()
        p2=None
        if self.state=='pending' and np.hypot(pose[0],pose[1])>=.12:
            candidate=self.source.reproject(rgb,depth,pose)
            if candidate is not None:
                feat,other=candidate;p2=float(model('relook')(feat[None])[0])
                if p2>=gates()['relook']:
                    self.verifier=other.verifier();self.anchor=tuple(pose);self.state='reanchored'
            if self.state=='pending' and 'corroboration' in gates():
                e,bb=self.verifier.evidence(rgb,depth,pose)
                second=corroborate(self.source,bb,e,rgb,depth,pose)
                if second is not None:
                    features,new_surface=second
                    confidence=float(model('corroboration')(features[None])[0])
                    if confidence>=gates()['corroboration']:
                        self.secondary=self.verifier
                        self.verifier=new_surface.verifier();self.anchor=tuple(pose);self.state='reanchored'
                        p2=confidence
        if self.state=='pending':
            out=empty()
            out.update(memory_state=self.state,p_initial=self.initial_confidence,p_relook=p2,reason='need_independent_second_view')
            return out
        rel=relative(pose,self.anchor)
        out=self.verifier.observe(rgb,depth,rel)
        if self.use_clarity and out['box'] is not None:
            out['p_clear']=float(self.clarity.predict(rgb,depth,out['box'],self.verifier,rel,out))
            out['accept']=out['p_clear']*out['p_same']>=thresholds()['threshold']
        if self.secondary is not None and out['box'] is not None:
            second=self.secondary.observe(rgb,depth,pose)
            if second['box'] is not None:
                a,b=out['box'],second['box']
                overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
                union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
                if overlap/max(1,union)>=.55:
                    if self.use_clarity:
                        second['p_clear']=float(self.clarity.predict(rgb,depth,b,self.secondary,pose,second))
                        second['accept']=second['p_clear']*second['p_same']>=thresholds()['threshold']
                    if second['p_clear']*second['p_same']>out['p_clear']*out['p_same']:out=second
        out.update(memory_state=self.state,p_initial=self.initial_confidence,p_relook=p2)
        return out
