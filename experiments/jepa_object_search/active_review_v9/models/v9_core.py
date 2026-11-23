"""Causal small target-grounding and active re-observation heads."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Local(nn.Module):
    def __init__(self, queries):
        super().__init__()
        self.query_count = queries
        self.net = nn.Sequential(nn.Linear(64 + queries, 96), nn.LayerNorm(96), nn.GELU(), nn.Linear(96, 2*queries))

    def forward(self, code, similarities):
        a,b = self.net(torch.cat([6*code, 20*(similarities-.20)],-1)).chunk(2,-1)
        v=(a+16*(similarities-.20)).sigmoid()
        return v,v*b.sigmoid()


def descriptors(visible, near, similarity, code, memory=None):
    """One row per view and query. Memory is a prior reported crop's visual signature."""
    # entry shapes [views,49,terms], [views,49,64] and optional [64] normalized crop.
    match=code.new_zeros((*code.shape[:-1],1)) if memory is None else (
        F.normalize(code,dim=-1) @ F.normalize(memory.flatten(),dim=0)).unsqueeze(-1)
    signals=[visible,near,similarity,visible*match]
    def summarize(s):
        s=s.reshape(len(s),7,7,-1)
        vals=[s.mean((1,2)),s.flatten(1,2).topk(4,dim=1).values.mean(1)]
        for low,high in ((0,2),(2,5),(5,7)):
            g=s[:,:,low:high].flatten(1,2)
            vals.append(g.topk(3,dim=1).values.mean(1))
        return torch.stack(vals,-1)
    return torch.cat([summarize(x) for x in signals],-1)


class ReviewHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(43,64),nn.LayerNorm(64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,3))

    def forward(self, current, previous, delta, has_memory):
        # delta is previous camera bin relative to present (positive=left).
        d=torch.as_tensor(delta,dtype=current.dtype,device=current.device)
        mem=torch.as_tensor(has_memory,dtype=current.dtype,device=current.device)
        tail=torch.stack(torch.broadcast_tensors(d,torch.abs(d),mem, current[...,0])[0:3],-1)
        return self.layers(torch.cat([current,previous-current,tail],-1))


def decide(prob, threshold):
    """Order = current true evidence, after left 30°, after right 30°."""
    p=np.asarray(prob)
    if threshold is not None and p[0]>=threshold:
        return "review"
    k=int(np.argmax(p[1:]))+1
    if p[k] >= .15 and p[k]-p[0]>.03:
        return ("left","right")[k-1]
    return "continue"


def field_labels(grid, eligible):
    """Per-instance proxy for identifiable and legally reportable evidence in one image."""
    pixels=grid.sum((-1,-2))
    interior=grid[:,:,:,1:6].sum((-1,-2))
    return ((pixels>=192)&(interior>=32)&((grid>=12).sum((-1,-2))>=2)&eligible[None,:])


def crop_box(grid):
    yy,xx=np.nonzero(grid>0)
    if not len(xx):return None
    return (int(xx.min()*256//7),int(yy.min()*256//7),
            int(math.ceil((xx.max()+1)*256/7)),int(math.ceil((yy.max()+1)*256/7)))
