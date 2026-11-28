"""Local, bounded path repair for an already chosen target instance and observation pose."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from path_geometry import NAMES

COL={name:i for i,name in enumerate(NAMES)}
SEED=1128


@dataclass
class ProposalSet:
    """Only actor inputs; no environment labels, collision outcomes or future images."""
    action:np.ndarray
    path:np.ndarray
    reach:np.ndarray
    threshold:np.ndarray
    suspect:np.ndarray
    side:np.ndarray
    decision:np.ndarray
    visual:np.ndarray
    old_joint:np.ndarray
    visible_if_reached:np.ndarray

    def __post_init__(self):
        n=len(self.action)
        self.length=(self.action!=0).sum(axis=1)
        if n<1 or np.any(self.length<1) or not np.isin(self.action,[0,1,2,3]).all():
            raise ValueError("invalid primitive plans")
        if np.any((self.action[:,:-1]==0)&(self.action[:,1:]!=0)):
            raise ValueError("nontrailing padding")
        if self.path.shape[:2]!=self.action.shape or self.path.shape[2]!=len(NAMES):
            raise ValueError("current-observation path features do not align with actions")
        for x in (self.reach,self.suspect,self.side,self.decision,self.visual,self.old_joint,self.visible_if_reached):
            if len(x)!=n:raise ValueError("unaligned candidate evidence")
        if not np.isin(self.decision,["accept","replan","reobserve"]).all():
            raise ValueError("unknown checker decision")
        self.threshold=np.broadcast_to(np.asarray(self.threshold),(n,))
        if not np.array_equal(self.decision=="accept",self.reach>=self.threshold):
            raise ValueError("checker thresholds disagree")
        self.ends=self.path[np.arange(n),self.length-1,:4]
        if not np.allclose(self.ends,self.ends[:1],atol=1e-3):
            raise ValueError("repairs must return to the same requested viewpoint")


def embedding_sketch(prediction,scene):
    """Fixed projection; applies identical pre-processing to unseen rooms."""
    d=prediction.shape[1]
    mat=np.random.default_rng(SEED).normal(0,1/np.sqrt(d),(d,24)).astype(np.float32)
    return ((prediction-scene)@mat*24).astype(np.float32)


def repair_features(s,previous,with_hint=True):
    rows=[]
    source=s.path[previous,:s.length[previous]]
    step=int(s.suspect[previous])
    is_block=s.decision[previous]=="replan"
    uncertain=s.decision[previous]=="reobserve"
    if not 0<=step<len(source):
        move=np.flatnonzero(source[:,COL["is_forward"]]>0)
        step=int(move[np.argmin(source[move,COL["path_coverage"]])]) if len(move) else 0
    point=source[step,:2]
    for i,n in enumerate(s.length):
        f=s.path[i,:n]
        moving=f[f[:,COL["is_forward"]]>0]
        if not len(moving):moving=f
        delta=moving[:,:2]-point
        distance=np.linalg.norm(delta,axis=1)
        nearest=int(distance.argmin())
        # Lateral coordinate is expressed in the rejected segment's heading.
        sideways=delta[nearest]@np.array([source[step,3],source[step,2]])
        hazard_side=s.side[previous]
        cue=[is_block,uncertain,step/max(1,len(source)),hazard_side,
             np.clip(distance.min(),0,3),np.mean(distance<.25),sideways,
             sideways*hazard_side,moving[nearest,COL["path_coverage"]],
             moving[nearest,COL["body_hit"]],source[step,COL["left_hit"]],source[step,COL["right_hit"]]]
        if not with_hint:cue=np.zeros(len(cue))
        base=[s.reach[i],s.visible_if_reached[i],s.old_joint[i],n/40,
              (s.action[i]==1).sum()/20, np.min(moving[:,0]),np.max(moving[:,0]),
              moving[:,COL["path_coverage"]].mean(),moving[:,COL["observed_clear"]].mean(),
              moving[:,COL["body_hit"]].mean(),moving[:,COL["occluded_fraction"]].mean()]
        rows.append(np.r_[base,s.visual[i],cue])
    return np.asarray(rows,np.float32)


class RepairHead(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(features,48),nn.LayerNorm(48),nn.GELU(),nn.Linear(48,24),nn.GELU())
        self.score=nn.Linear(24,1)
        self.visual_delta=nn.Linear(24,24)

    def forward(self,x):
        return self.score(self.net(x)).squeeze(-1)


class RepairScorer:
    def __init__(self,path):
        c=torch.load(path,map_location="cpu",weights_only=True)
        self.mean=c["mean"].numpy();self.scale=c["scale"].numpy()
        self.net=RepairHead(len(self.mean)).eval()
        self.net.load_state_dict(c["weights"])
        self.hints=bool(c["hints"])
        self.vision=bool(c["vision"])

    @torch.inference_mode()
    def score(self,s,previous):
        x=repair_features(s,previous,self.hints)
        if not self.vision:x[:,1:3]=0;x[:,11:35]=0
        return self.net(torch.from_numpy(np.clip((x-self.mean)/self.scale,-7,7))).numpy()


def decide(s,initial,budget,arm,scorer=None,max_checks=3):
    """Return actions only for a checker-cleared proposal, or an observation request."""
    if not 0<=initial<len(s.length) or max_checks<1:raise ValueError("invalid initial plan or allowance")
    actions=np.flatnonzero(s.length<=budget)
    if initial not in actions:raise ValueError("initial exceeds physical action budget")
    def accept(i,history):
        return dict(status="execute",index=int(i),actions=s.action[i,:s.length[i]].tolist(),history=[int(k) for k in history])
    if arm=="original":
        return accept(initial,[initial])
    if arm=="one_shot":
        # Full independent marginal predictions for all candidates, no rejection-conditioned retry.
        allowed=actions[s.decision[actions]=="accept"]
        if len(allowed):
            best=max(allowed,key=lambda i:float(s.reach[i]*s.visible_if_reached[i]))
            return accept(best,[best])
        return dict(status="request_observation",index=None,actions=[],history=[])
    current=initial
    history=[]
    while True:
        history.append(int(current))
        if s.decision[current]=="accept":return accept(current,history)
        remaining=np.array([i for i in actions if i not in history],dtype=int)
        if not len(remaining) or len(history)>=max_checks or arm=="gate_only":
            return dict(status="request_observation",index=None,actions=[],history=history)
        if arm=="jepa_order":
            score=s.old_joint
        elif arm=="independent":
            score=s.reach*s.visible_if_reached
        elif arm in ("trained","without_hint","geometry_only","mask_at_test"):
            score=scorer.score(s,current)
        else:raise ValueError(arm)
        current=remaining[np.argmax(score[remaining])]
