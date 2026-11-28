"""Observed-depth and action-only chassis feasibility. No instance labels or future observations."""
from pathlib import Path
import joblib
import numpy as np
from path_geometry import NAMES


def check_inputs(action,path):
    a=np.asarray(action)
    x=np.asarray(path,dtype=np.float32)
    if a.ndim!=2 or x.shape!=(*a.shape,len(NAMES)) or not len(a) or not a.shape[1]:
        raise ValueError("expected matching batches of primitive plans and per-action observations")
    if not np.isin(a,[0,1,2,3]).all() or not np.isfinite(x).all() or ((a==0).all(axis=1)).any():
        raise ValueError("invalid action or depth feature")
    if ((a[:,:-1]==0)&(a[:,1:]!=0)).any() or not np.array_equal(x[:,:,4][a>0],(a==1)[a>0]):
        raise ValueError("action padding or feature plan differs")
    return a,x


def summarize(action,path):
    rows=[]
    for acts,x in zip(action,path):
        move=acts==1
        real=x[acts>0]
        forward=x[move]
        no_forward=not len(forward)
        if no_forward:forward=np.zeros((1,len(NAMES)),np.float32)
        metrics=forward[:,5:]
        length=len(real)
        endpoint=real[-1,:2]
        route=[int(move.sum()),length,np.linalg.norm(endpoint),np.linalg.norm(real[:,:2],axis=1).max(),
               float((acts==2).sum()),float((acts==3).sum()),np.sum(forward[:,5]<.5)]
        # A path fails if even one segment is blocked; preserve extrema and absolute risky lengths.
        rows.append(np.concatenate((route,metrics.min(0),metrics.mean(0),metrics.max(0),
                                    (forward[:,13:16]>.5).sum(0),
                                    np.mean(forward[:,8:11]>.2,axis=0))))
    return np.asarray(rows,np.float32)


def per_step(acts,features):
    progress=np.cumsum(acts==1,axis=1)
    return np.concatenate([features,progress[:,:,None],np.linalg.norm(features[:,:,:2],axis=2)[:,:,None]],axis=2)


class PathChecker:
    def __init__(self,models):
        self.models=models
        if list(models["names"])!=list(NAMES):raise ValueError("feature schema differs from training")

    @classmethod
    def load(cls,path):
        return cls(joblib.load(Path(path)))

    def predict(self,action,path,*,threshold=None):
        a,x=check_inputs(action,path)
        route=self.models["route"]
        score=route.predict_proba(summarize(a,x))[:,1]
        any_move=(a==1).any(axis=1)
        score[~any_move]=1.
        threshold=self.models["threshold"] if threshold is None else float(threshold)
        if not np.isfinite(threshold) or threshold<0 or threshold>np.nextafter(1.,2.):
            raise ValueError("invalid calibrated threshold")
        accept=score>=threshold
        suspect=np.full(len(a),-1,int)
        side=np.zeros(len(a),int)
        decision=np.where(accept,"accept","reobserve")
        feedback=np.where(accept,"clear","uncertain_path").astype("<U20")
        # The learned step likelihood localizes a predicted blockage; require a measured surface
        # before suggesting a side. A gap in depth support alone only requests another observation.
        active=(~accept[:,None])&(a==1)
        local=np.zeros(a.shape,np.float32)
        if active.any():
            local[active]=self.models["local"].predict_proba(per_step(a,x)[active])[:,1]
        for i in np.flatnonzero(~accept):
            occupied=(a[i]==1)&(x[i,:,13]>.5)
            if not occupied.any():continue
            idx=np.flatnonzero(occupied)
            t=int(idx[np.argmax(local[i,idx])])
            suspect[i]=t
            side[i]=int(np.sign(x[i,t,15]-x[i,t,14]))
            decision[i]="replan"
            feedback[i]="observed_obstacle"
        return dict(p_reach=score,threshold=np.full(len(a),threshold),decision=decision,
                    feedback=feedback,suspect_step=suspect,suspect_side=side)
