"""Goal-relative semantic interaction for short action-conditioned visual forecasts."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SemanticFork(nn.Module):
    def __init__(self,center,basis,anchors):
        super().__init__()
        self.register_buffer("center",center.clone())
        self.register_buffer("basis",basis.clone())
        self.register_buffer("anchors",F.normalize(anchors.clone(),dim=-1))
        dim=basis.shape[1]
        self.residual=nn.Sequential(nn.Linear(2*dim+5,48),nn.GELU(),nn.Linear(48,dim))
        self.compatibility=nn.Sequential(nn.Linear(2*dim+2*len(anchors),48),nn.GELU(),
                                        nn.Linear(48,len(anchors)))
        nn.init.zeros_(self.compatibility[-1].weight)
        nn.init.zeros_(self.compatibility[-1].bias)

    def forward(self,forecast,nearest,motion,query,observed_only=False):
        movement=motion[:,:,None].expand(-1,-1,3,-1)
        base=(forecast-self.center) @ self.basis
        delta=(forecast-nearest) @ self.basis
        if observed_only:
            predicted=nearest
        else:
            residual=self.residual(torch.cat([base,delta,movement],-1))
            predicted=F.normalize(forecast+.12*residual.tanh()@self.basis.T,dim=-1)
        current=20*(nearest @ self.anchors.T-.22)
        future=20*(predicted @ self.anchors.T-.22)
        z=(predicted-self.center) @ self.basis
        dz=(predicted-nearest) @ self.basis
        features=torch.cat([6*z,6*dz,future,(future-current)],-1)
        learned=self.compatibility(features)
        # All candidate-common target evidence in the semantic term is explicitly
        # eliminated with the *same fixed* bank also used on an unseen free-text goal.
        bank=torch.logsumexp((future+learned).flatten(1,2),dim=1)-math.log(6)
        weights=(20*query@self.anchors.T).softmax(-1)
        anchor_weights=(20*self.anchors@self.anchors.T).softmax(-1)
        # Direct free-text alignment preserves appearance/attributes outside the bank.
        clip=20*(predicted@query.T-.22)
        direct=torch.logsumexp(clip.flatten(1,2),dim=1)-math.log(6)
        anchor_direct=torch.logsumexp(future.flatten(1,2),dim=1)-math.log(6)
        offset=bank-anchor_direct
        evidence=direct+offset@weights.T
        common=(anchor_direct+offset@anchor_weights.T).mean(1,keepdim=True)
        return evidence-common,predicted


def displacement(movement):
    # Per-horizon nominal translational displacement / maximum reachable length.
    delta=movement[:,-1,:2].norm(dim=1)
    fraction=delta/movement[:,-1,4].clamp_min(1e-8)
    return fraction.clamp(0,1)


def compose(semantic,motion,strength):
    prior=strength*(2*displacement(motion)-1)
    score=(semantic+prior[:,None]).sigmoid()
    null=prior.sigmoid()[:,None].expand_as(score)
    return score,null


def restore(path):
    saved=torch.load(path,map_location="cpu",weights_only=True)
    weights=saved["model"]
    head=SemanticFork(weights["center"],weights["basis"],weights["anchors"])
    head.load_state_dict(weights)
    return head.eval(),saved["strength"]
