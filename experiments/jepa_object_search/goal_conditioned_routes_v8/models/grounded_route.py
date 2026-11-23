"""Causal, goal-conditioned short-route verification using dense observed visual evidence."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def cell_xy(reference):
    axis=torch.linspace(-1+1/7,1-1/7,7,device=reference.device,dtype=reference.dtype)
    yy,xx=torch.meshgrid(axis,axis,indexing="ij")
    return torch.stack([xx.flatten(),yy.flatten()],-1)


class LocalGrounder(nn.Module):
    """Learn presence and conditional nearness of an object in a real observation."""
    def __init__(self,anchors,dim=64):
        super().__init__()
        self.register_buffer("anchors",F.normalize(anchors,dim=-1))
        self.context=nn.Sequential(nn.Linear(dim+len(anchors)+2,72),nn.GELU(),
                                   nn.Linear(72,2*len(anchors)))
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def logits(self,code,anchor_sim,query_sim,text):
        xy=cell_xy(code).expand(len(code),-1,-1)
        fields=torch.cat([code*6,20*(anchor_sim-.20),xy],-1)
        present,near=self.context(fields).chunk(2,-1)
        weights=(24*text@self.anchors.T).softmax(-1)
        return 20*(query_sim-.20)-3+present@weights.T,-1+near@weights.T

    def forward(self,*args):
        visible,near=self.logits(*args)
        probability=visible.sigmoid()
        return probability,probability*near.sigmoid()


def route_geometry(actions,step=.25,turn=math.pi/6):
    yaw=((actions==1).float()-(actions==2).float()).cumsum(1)*turn
    f=(actions==0).float()
    xx=(f*yaw.cos()*step).cumsum(1);yy=(f*yaw.sin()*step).cumsum(1)
    pick=[actions.shape[1]//2-1,actions.shape[1]-1]
    density=f.cumsum(1)[:,pick]/f.new_tensor([j+1 for j in pick])
    disp=(xx[:,pick].square()+yy[:,pick].square()).sqrt()
    length=f.new_tensor([j+1 for j in pick]).expand_as(disp)
    geom=torch.stack([xx[:,pick]/6,yy[:,pick]/6,yaw[:,pick].sin(),yaw[:,pick].cos(),
                      density,disp/(length*step),length/24],-1)
    return xx[:,pick],yy[:,pick],yaw[:,pick],geom


def evidence_fields(visible,near,heading,actions,step=.25,turn=math.pi/6):
    """Project only observed bearings under explicit distance hypotheses, never oracle depth."""
    x,y,yaw,geom=route_geometry(actions,step,turn)
    xy=cell_xy(visible)
    bearing=heading[:,None]-torch.atan(xy[:,0])[None]
    # Ranges are hypotheses, not estimated object poses.
    radii=visible.new_tensor([.75,1.5,3.,5.])
    anchor_x=bearing.flatten().cos()[None,:]*radii[:,None]
    anchor_y=bearing.flatten().sin()[None,:]*radii[:,None]
    dx=anchor_x[None,None]-x[:,:,None,None]
    dy=anchor_y[None,None]-y[:,:,None,None]
    dist=(dx.square()+dy.square()).sqrt()
    face=(dx*yaw.cos()[:,:,None,None]+dy*yaw.sin()[:,:,None,None])/dist.clamp_min(.001)
    approach=torch.exp(-.5*((dist-1.)/1.).square())
    in_view=torch.sigmoid(5*(face-.45))
    v=visible.flatten(0,1);n=near.flatten(0,1)
    # Max-pool over valid observed cells, retaining target dependence and range hypothesis.
    def pool(weights,signal):
        return (weights[...,None]*signal[None,None,None]).amax(3).transpose(2,3)
    arrival=pool(approach*in_view,v); seeing=pool(in_view,v); occupied=pool(approach*in_view,n)
    nearest=pool(approach,v)
    summary=torch.stack([v.amax(0),n.amax(0),v.mean(0),n.mean(0)],-1)
    return torch.cat([arrival,seeing,occupied,nearest,summary.expand(len(x),2,-1,-1)],-1),geom


def jepa_fields(future,nearest,text,anchors):
    target=20*(future@text.T-.20)
    aligned=20*((future-nearest)@text.T)
    rel=target-20*(future@anchors.T-.20).mean(-1,keepdim=True)
    return torch.cat([z.permute(0,1,3,2) for z in (target,aligned,rel)],-1)


class RouteVerifier(nn.Module):
    """Read out report opportunities, near-target visibility and route progress."""
    def __init__(self,mode):
        super().__init__()
        if mode not in ("geometry","observed","jepa","joint"):raise ValueError(mode)
        self.mode=mode
        # Expanding masked inputs to the same width gives each comparison equal capacity.
        self.body=nn.Sequential(nn.Linear(36,64),nn.LayerNorm(64),nn.GELU(),nn.Linear(64,32),nn.GELU())
        self.tasks=nn.Linear(32,3)
        nn.init.zeros_(self.tasks.bias)

    def forward(self,dense,forecast,geom):
        a=dense if self.mode in ("observed","joint") else torch.zeros_like(dense)
        b=forecast if self.mode in ("jepa","joint") else torch.zeros_like(forecast)
        inputs=torch.cat([a,b,geom[:,:,None].expand(-1,-1,a.shape[2],-1)],-1)
        z=self.tasks(self.body(inputs))
        opportunity=z[...,0]+.35*z[...,1]
        score=torch.logsumexp(opportunity,dim=1)-math.log(2)
        return score,z[...,1],z[...,2].mean(2)
