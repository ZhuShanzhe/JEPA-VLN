"""Learn which environments lie along a route and rank them for an object goal."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def dependencies(project):
    sys.path.insert(0,str(project/'jepa_route_v12/models'))
    from goal_ranker import previous,load_prior
    return previous(project),load_prior


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Reader(nn.Module):
    """An observation reader trained on RGB at the actual camera location."""
    def __init__(self, dim, k):
        super().__init__()
        self.read=nn.Sequential(nn.Linear(3*dim,96),nn.LayerNorm(96),nn.GELU(),nn.Linear(96,k))

    def forward(self,views):
        return self.read(views.flatten(-2))


class RouteHead(nn.Module):
    def __init__(self,dim,k):
        super().__init__()
        self.dim=dim
        self.context=nn.Sequential(nn.Linear(dim*3+8,144),nn.LayerNorm(144),nn.GELU(),nn.Linear(144,112),nn.GELU())
        self.predict=nn.Linear(112,dim)
        self.room=nn.Linear(112+dim,k)
        self.goal=nn.Linear(dim,32)
        self.utility=nn.Sequential(nn.Linear(112+32+2,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,x,text,prior):
        h=self.context(x)
        pred=F.normalize(x[...,:self.dim]+self.predict(h),dim=-1)
        rooms=self.room(torch.cat([h,pred],dim=-1))
        env=rooms.softmax(-1)
        compatibility=(env*prior[:,None,None]).sum(-1)
        visual=(pred*text[:,None,None]).sum(-1)
        target=self.goal(text)[:,None,None].expand(*h.shape[:-1],32)
        gain=self.utility(torch.cat([h,target,compatibility[...,None],visual[...,None]],dim=-1))[...,0]
        gain=gain+2*(compatibility-.5)
        logits=torch.logsumexp(gain,dim=-1)-np.log(gain.shape[-1])
        return logits,rooms,pred


def prior_vector(raw,word):
    values=raw.get('p_target_given_room',{}).get(word)
    if values is not None:
        return np.asarray(values,np.float32)
    p=np.asarray(raw['p_room_prior'],np.float32)
    given=np.asarray(raw['p_room_given_target'].get(word,p),np.float32)
    lift=given/np.maximum(p,1e-6)
    return lift/(1.+lift)


class EnvironmentRoutes:
    def __init__(self,project,seed=11):
        self.project=Path(project); d=self.project/'jepa_env_v13/models'
        self.old,load_prior=dependencies(self.project)
        from encode import Encoder
        self.encoder=Encoder(self.project);self.device=self.encoder.device
        self.config=json.loads((d/'selection.json').read_text())
        self.raw=json.loads((d/'prior.json').read_text())
        self.room_names=self.raw['room_names']
        self.rooms=self.room_names
        k=len(self.rooms);dim=self.encoder.proj.shape[1]
        self.base=load_prior(self.old,self.project,seed,'jepa',self.device)
        self.reader=Reader(dim,k).to(self.device)
        self.heads={arm:RouteHead(dim,k).to(self.device) for arm in ('direct','jepa')}
        for arm,net in [('reader',self.reader),*self.heads.items()]:
            entry=self.config[arm]
            code='reader' if arm=='reader' else f'{arm}_{seed}_{entry["step"]}'
            path=d/'weights'/(code+'.pt')
            if digest(path)!=entry['hashes'][code]:
                raise ValueError(f'weight hash differs for {code}')
            net.load_state_dict(torch.load(path,map_location=self.device,weights_only=True));net.eval()

    def tensor(self,v):
        return torch.as_tensor(v,device=self.device,dtype=torch.float32)

    @torch.inference_mode()
    def cached(self,ctx,text,plans,target):
        plans=np.asarray(plans)
        x=self.tensor(self.old.route_inputs(np.asarray(ctx,dtype=np.float32),plans))[None]
        t=self.tensor(text).reshape(1,-1);prior=self.tensor(prior_vector(self.raw,target))[None]
        base=self.base(x,t)[0].sigmoid()[0]
        result=dict(v11=base.cpu().numpy(),prior=prior[0].cpu().numpy())
        # Include intermediate route views in the semantic forecast (four equal taps).
        for arm,net in self.heads.items():
            score,room,_=net(x,t,prior)
            dist=room.softmax(-1)[0].mean(1)
            mix=self.config[arm]['mix']
            result[arm]=((1-mix)*base+mix*score.sigmoid()[0]).cpu().numpy()
            result['env_pred_'+arm]=dist.cpu().numpy()
            result['env_steps_'+arm]=room.softmax(-1)[0].cpu().numpy()
            result['semantic_'+arm]=(dist@prior[0]).cpu().numpy()
        result['env_pred']=result['env_pred_jepa']
        # Weak no-lookahead comparison: departure camera orientation closest to
        # the planned bearing, chosen from only the three observed frames.
        views=self.reader(self.tensor(ctx)).softmax(-1)
        ang=np.cumsum((plans==1).astype(float)-(plans==2),axis=1)*np.pi/6
        fwd=(plans==0)*.25
        dx=np.cumsum(fwd*np.cos(ang),axis=1);dy=np.cumsum(fwd*np.sin(ang),axis=1)
        bearing=np.where(np.hypot(dx,dy)>.05,np.arctan2(dy,dx),ang)
        idx=np.argmax(np.cos(bearing[:,:,None]-np.deg2rad([-30.,0.,30.])),axis=-1)
        result['env_pred_copy']=views[torch.as_tensor(idx[:,5::6],device=self.device)].mean(1).cpu().numpy()
        return result

    @torch.inference_mode()
    def live(self,rgb,target,plans):
        t=self.encoder.words([target])
        ctx,cue=self.encoder.encode(np.asarray(rgb),t)
        res=self.cached(ctx,F.normalize(t@self.encoder.proj,dim=-1).cpu().numpy()[0],np.asarray(plans),target)
        res['cue']=cue[:,0]
        if len(self.encoder.cache)>600:self.encoder.cache.clear()
        return res

    @torch.inference_mode()
    def observe_batch(self,rgb,target):
        feats,cue=self.encoder.encode(np.asarray(rgb),self.encoder.words([target]))
        prob=self.reader(self.tensor(feats)).softmax(-1).cpu().numpy()
        if len(self.encoder.cache)>600:self.encoder.cache.clear()
        return prob,cue[:,0]

    def observed(self,rgb):
        return self.observe_batch(rgb,'chair.n.01')[0]
