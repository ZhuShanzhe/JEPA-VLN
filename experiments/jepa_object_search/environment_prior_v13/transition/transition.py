"""Predict changes of room appearance from departure images and nominal motion."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


def headings(plans):
    turn=(plans==1).astype(float)-(plans==2)
    yaw=np.cumsum(turn,axis=1)*np.pi/6
    taps=np.arange(5,24,6)
    ahead=(plans==0)*.25
    x=np.cumsum(ahead*np.cos(yaw),axis=1)
    z=np.cumsum(ahead*np.sin(yaw),axis=1)
    bearing=np.where(np.hypot(x,z)>.05,np.arctan2(z,x),yaw)
    pick=np.cos(bearing[:,:,None]-np.deg2rad([-30.,0,30])).argmax(-1)[:,taps]
    turning=np.stack([np.cumsum(abs(turn),axis=1)[:,taps]/24,
             np.sin(yaw[:,taps]),np.cos(yaw[:,taps]),np.hypot(x,z)[:,taps]/6],axis=-1)
    return pick,turning


class ChangeNet(nn.Module):
    def __init__(self,dim,rooms,residual=False):
        super().__init__()
        self.residual=residual
        self.encoder=nn.Sequential(nn.Linear(dim,72),nn.LayerNorm(72),nn.GELU(),
                                   nn.Dropout(.10),nn.Linear(72,48),nn.GELU())
        self.cross=nn.Linear(48,1)
        self.destination=nn.Linear(48,rooms)
    def forward(self,x,departure,original=None):
        h=self.encoder(x)
        logit=self.cross(h).squeeze(-1)
        route=self.destination(h)
        moved=logit.sigmoid().unsqueeze(-1)
        if self.residual:
            logits=original.clamp_min(1e-6).log()+moved*(route-route.mean(-1,keepdim=True)).tanh()*1.25
            return logits.softmax(-1),logit,route
        return (1-moved)*departure+moved*route.softmax(-1),logit,route


@torch.inference_mode()
def features(base,ctx,plans):
    x=base.tensor(base.old.route_inputs(np.asarray(ctx,dtype=np.float32),plans))
    h=base.heads['jepa'].context(x)
    pred=torch.nn.functional.normalize(x[...,:base.encoder.proj.shape[1]]+base.heads['jepa'].predict(h),dim=-1)
    original=base.heads['jepa'].room(torch.cat([h,pred],-1)).softmax(-1)
    idx,extra=headings(plans)
    scan=base.reader(base.tensor(ctx)).softmax(-1)
    departure=scan[torch.as_tensor(idx,device=base.device)]
    distribution=scan.flatten().reshape(1,1,-1).expand(3,4,-1)
    return (torch.cat([h,departure,original,distribution,base.tensor(extra),x[...,-8:]],dim=-1),
            departure,original,idx)


class TransitionPredictor:
    def __init__(self,project,base,seed,residual=False):
        self.base=base
        d=Path(project)/'jepa_env_v13/transition'
        if residual:d=d/'residual'
        spec=json.loads((d/'selection.json').read_text())
        self.mix=spec['selected']['mix']
        ck=d/f'seed_{seed}_step_{spec["selected"]["step"]}.pt'
        if hashlib.sha256(ck.read_bytes()).hexdigest()!=spec['sha256'][str(seed)]:
            raise ValueError('checkpoint hash differs')
        self.net=ChangeNet(spec['input_dim'],len(base.room_names),residual=residual).to(base.device).eval()
        self.net.load_state_dict(torch.load(ck,map_location=base.device,weights_only=True))

    @torch.inference_mode()
    def forecast(self,ctx,plans):
        x,departure,original,_=features(self.base,ctx,plans)
        p=self.net(x,departure,original)[0]
        return ((1-self.mix)*original+self.mix*p).cpu().numpy()
