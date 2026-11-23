"""Causal route features and offline grouping of training examples."""
import hashlib
import math

import torch
from torch.nn import functional as F

from grounded_route import evidence_fields, jepa_fields
from route_heads import route_logits, view_inputs

H=(4,12,24)

@torch.no_grad()
def forecast_inputs(heads, mode, observed, headings, actions, step=.25, turn=math.pi/6):
    nearest, scene, motion = view_inputs(observed, headings, actions, step, turn)
    if mode == "full":
        predicted = heads[0](nearest, scene, motion)
    elif mode == "sparse":
        predicted, nearest, _ = heads[1](observed, headings, actions, step, turn)
    else:
        raise ValueError(f"unknown view mode {mode!r}")
    return predicted, nearest, motion


def mask_bins(sid,n):
    if n==12:return list(range(n))
    sign=1 if hashlib.sha256(("23|"+sid).encode()).digest()[0]&1 else -1
    return sorted((sign*i)%12 for i in range(n))


@torch.no_grad()
def build_features(heads,ground,s,anchors):
    # Explicit argument list excludes grid labels, true collision, pose, and future RGB.
    v,n=ground(s["code"],s["anchor_sim"],s["anchor_sim"][:,:,s["ids"]],s["text"])
    average=anchors.mean(0,keepdim=True)
    neutral=F.normalize(average,dim=-1)
    neutral_sim=s["anchor_sim"].mean(-1,keepdim=True)/average.norm().clamp_min(1.e-8)
    nv,nn=ground(s["code"],s["anchor_sim"],neutral_sim,neutral)
    groups=[]
    for hi,h in enumerate(H):
        a=s["actions"][:,:h]
        g=dict(sid=s["sid"],house=s["house"],family=s["family"],query=s["query"],domain=s["domain"],
               y=s["gains"][:,hi],mask=(a[:,None]!=a[None,:]).any(-1),modes=[],null_modes=[],h=h)
        counts=(a==0).sum(1)
        yaw=((a==1).long()-(a==2).long()).cumsum(1)*math.pi/6
        dx=((a==0)*.25*yaw.cos()).sum(1);dy=((a==0)*.25*yaw.sin()).sum(1)
        dis=(dx.square()+dy.square()).sqrt()
        sep=(a[:,None]!=a[None,:]).any(-1)
        g["matched"]=sep & ((counts[:,None]-counts[None,:]).abs()<=1) & ((dis[:,None]-dis[None,:]).abs()<=.501)
        swapped=torch.where(a==0,a,3-a)
        g["mirror"]=sep & (swapped[:,None]==a[None,:]).all(-1)
        g["near"]=(s["teacher_near"][:,[s["samples"].index(t) for t in (h//2,h)]]>=64).float()
        actual=s["teacher_actions"][:,:h]
        g["forward"]=torch.stack([(actual[:,:t]==0).any(1) for t in (h//2,h)],1)
        g["progress"]=torch.stack([1.-(s["collision"][:,:t]&(actual[:,:t]==0)).sum(1)/
                                  (actual[:,:t]==0).sum(1).float().clamp_min(1) for t in (h//2,h)],1)
        for nm in (12,1,2,3):
            bins=mask_bins(s["sid"],nm)
            heading=torch.tensor(bins).float()*math.pi/6
            p,nearest,motion=forecast_inputs(heads,"full" if nm==12 else "sparse",s["ctx"][bins],heading,a)
            dense,geom=evidence_fields(v[bins],n[bins],heading,a)
            J=jepa_fields(p,nearest,s["text"],anchors);N=jepa_fields(nearest,nearest,s["text"],anchors)
            g["modes"].append((dense,J,geom,N,route_logits(p,s["text"]).sigmoid()))
            ng,_=evidence_fields(nv[bins],nn[bins],heading,a)
            g["null_modes"].append((ng,jepa_fields(p,nearest,neutral,anchors)))
        groups.append(g)
    return groups
