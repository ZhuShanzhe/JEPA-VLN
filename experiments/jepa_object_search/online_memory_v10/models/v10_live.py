"""Causal, episode-local route scoring. All coordinates are relative odometry in metres."""
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np

FEATURES=("prior","geometry","novelty","clearance","target_trace")
DEFAULT={"prior":.18,"geometry":.06,"novelty":.11,"clearance":.10,"target_trace":.12}

def _vec(value):
    a=np.asarray(value,dtype=float)
    if a.shape!=(2,) or not np.isfinite(a).all():raise ValueError("expected finite 2-vector")
    return a

def _unit(value):
    a=_vec(value);norm=np.linalg.norm(a)
    if not .95 < norm < 1.05:raise ValueError("heading must be a unit vector")
    return a/norm

def _step(v):
    if not isinstance(v,(int,np.integer)) or v<0:raise ValueError("nonnegative integral step required")
    return int(v)

def nominal_paths(plans,xz,head):
    a=np.asarray(plans)
    if a.ndim!=2 or not a.size or not np.isin(a,[0,1,2]).all():raise ValueError("invalid plans")
    yaw=math.atan2(head[1],head[0])+np.cumsum((a==1).astype(float)-(a==2),axis=1)*math.pi/6
    direction=np.stack([np.cos(yaw),np.sin(yaw)],-1)
    return xz+np.cumsum(direction*(a==0)[:,:,None]*.25,axis=1),direction

def prior_transform(p):
    v=np.asarray(p,float)
    if not np.isfinite(v).all() or np.any(v<0) or np.any(v>1):raise ValueError("invalid probability")
    return v/(v+.12)

class RouteMemory:
    """Weights stay frozen. Saved state consists only of quantized location/bearing evidence."""
    def __init__(self,weights_path):
        raw=Path(weights_path).read_bytes();self.fingerprint=hashlib.sha256(raw).hexdigest()
        self.weights=json.loads(raw)["weights"]
        if set(self.weights)!=set(FEATURES) or not np.isfinite(list(self.weights.values())).all():raise ValueError("invalid weights")
        self.reset()

    def reset(self):
        self.step=-1; self.last=None; self.history={}

    def observe(self,xz,step,heading_xz,visual_score=None,collided=False):
        point=_vec(xz);direction=_unit(heading_xz);step=_step(step)
        if visual_score is not None:
            visual_score=float(visual_score)
            if not math.isfinite(visual_score) or not 0<=visual_score<=1:raise ValueError("invalid visual evidence")
        event=[point.tolist(),direction.tolist(),visual_score,bool(collided)]
        if step==self.step:
            if event!=self.last:raise ValueError("different observations at the same step")
            return
        if step<self.step:raise ValueError("out-of-order observation")
        if self.last is not None:
            prev=np.asarray(self.last[0]);move=point-prev
            if np.linalg.norm(move)>.10:
                for v in self.history.values():
                    if v[6]>=0 and np.linalg.norm(prev-v[0])<.6 and np.dot(move,v[1])>.10:
                        v[6]=-1
        self.step=step;self.last=event
        key=self._key(point,direction)
        old=self.history.get(key)
        # New conflicting evidence for the same place and bearing replaces the old cue.
        # Missing visual evidence is distinct from having looked and seen little.
        vis_t=step if visual_score is not None else (-1 if old is None else old[5])
        cue=visual_score if visual_score is not None else (None if old is None else old[3])
        last_block=step if collided else (-1 if old is None else old[6])
        self.history[key]=[point.tolist(),direction.tolist(),step,cue,bool(collided),vis_t,last_block]
        if step%24==0:self.history={k:v for k,v in self.history.items() if step-v[2]<240}

    @staticmethod
    def _key(p,d):
        return '/'.join(map(str,[*np.floor(p/.5+.5).astype(int),round(math.atan2(d[1],d[0])/(math.pi/6))%12]))

    def save(self,path):
        p=Path(path)
        record=dict(schema=1,weights_sha256=self.fingerprint,step=self.step,last=self.last,history=self.history)
        tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(record,allow_nan=False));os.replace(tmp,p)

    def load(self,path):
        data=json.loads(Path(path).read_text())
        if data.get("weights_sha256")!=self.fingerprint or data.get("schema")!=1:raise ValueError("incompatible memory snapshot")
        self.step=data["step"];self.last=data["last"];self.history=data["history"]

    def features(self,plans,xz,heading_xz,step):
        step=_step(step)
        if step<self.step:raise ValueError("future observation in memory")
        pos=_vec(xz);direction=_unit(heading_xz)
        points,bearings=nominal_paths(plans,pos,direction)
        # Evaluate quarter, half and final positions (average within a candidate).
        n=points.shape[1];idx=np.unique([max(0,n//4-1),max(0,n//2-1),n-1]);points=points[:,idx];bearings=bearings[:,idx]
        records=list(self.history.values())
        if not records:return np.zeros((len(points),3))
        loc=np.array([v[0] for v in records]);look=np.array([v[1] for v in records])
        seen=np.array([v[2] for v in records]);vtime=np.array([v[5] for v in records]);block=np.array([v[6] for v in records])
        age=lambda t,half:np.where(t>=0,np.exp2(-np.maximum(0,step-t)/half),0.)
        rec=age(seen,65.)
        # Trace coverage is spatial, therefore turning in place cannot create explored floor.
        dist2=((points[:,:,None,:]-loc)**2).sum(-1)
        coverage=(np.exp(-dist2/(2*.75**2))*rec).max(-1).mean(-1)
        # A failed forward step provides local, directional, time-limited obstacle evidence.
        block_loc=loc+.55*look
        facing=np.maximum(0,np.einsum("chd,rd->chr",bearings,look))
        gap=((points[:,:,None,:]-block_loc)**2).sum(-1)
        hazard=(np.exp(-gap/(2*.70**2))*facing**2*age(block,18.)).max(-1).mean(-1)
        # A bearing is a visibility hypothesis, not a known object position. Optional
        # visual evidence follows that ray and weakens as newer observations accumulate.
        xy=points[:,:,None,:]-loc
        along=(xy*look).sum(-1)
        perp=np.maximum(0,dist2-along**2)
        area=np.exp(-.5*((along-1.6)/1.25)**2-.5*perp/.85**2)
        area*=np.clip((along+.25)/.8,0,1)*age(vtime,24.)
        cue=np.array([0. if v[3] is None else float(v[3])-.32 for v in records])
        cue_pos=np.maximum(cue,0.);cue_neg=np.maximum(-cue,0.)
        trace=((area*cue_pos).max(-1)-(area*cue_neg).max(-1)).mean(-1)
        return np.stack([1-coverage,-hazard,trace],axis=-1)

    def rank(self,plans,xz,heading_xz,step,prior,observed,geometry,*,arm="full"):
        if arm not in ("full","fixed","observed","geometry","v8"):raise ValueError("unsupported arm")
        p=prior_transform(prior);o=prior_transform(observed);g=prior_transform(geometry)
        if p.shape!=(len(plans),) or o.shape!=p.shape or g.shape!=p.shape:raise ValueError("mismatched candidate scores")
        mem=self.features(plans,xz,heading_xz,step)
        # Use a constant 0.5 offset only to preserve callers' probability-like thresholds.
        w=self.weights
        score=.5+w["geometry"]*g
        if arm!="geometry":score=score+w["prior"]*(o if arm=="observed" else p)
        if arm!="fixed":score=score+mem[:,0]*w["novelty"]+mem[:,1]*w["clearance"]
        if arm in ("full","observed"):score=score+mem[:,2]*w["target_trace"]
        if arm=="v8":score=np.asarray(prior,float)
        return dict(scores=score.tolist(),memory_features=mem.tolist(),arm=arm)
