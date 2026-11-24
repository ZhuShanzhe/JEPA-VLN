"""Run a calibrated frozen-V11 plus bounded correction on live RGB and online memory."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bounded_routes import ForegroundResidual, digest, load_prior, previous


class BoundedSelector:
    def __init__(self,project,arm='rv_jepa',seed=11):
        project=Path(project)
        self.old=previous(project)
        from encode import Encoder
        self.encoder=Encoder(project)
        self.device=self.encoder.device
        self.base=load_prior(self.old,project,seed,'jepa',self.device)
        d=project/'jepa_route_v12/refinement'
        settings=json.loads((d/'selection.json').read_text())['arms'][arm]
        self.cap=settings['cap']
        self.residual=None
        if self.cap>0:
            path=d/'weights'/f'{arm}_{seed}_{settings["step"]}.pt'
            if digest(path)!=settings['hashes'][str(seed)]:
                raise ValueError('residual weight digest mismatch')
            ck=torch.load(path,map_location=self.device,weights_only=True)
            self.residual=ForegroundResidual(len(ck['mean']),ck['mean']).to(self.device)
            self.residual.load_state_dict(ck)
            self.residual.eval()

    @torch.inference_mode()
    def scores(self,rgb,target,plans,episode_memory,time=2,xy=(0.,0.),facing=(1.,0.),first_observation=False):
        views=np.asarray(rgb);plans=np.asarray(plans)
        if views.dtype!=np.uint8 or views.ndim!=4 or views.shape[0]!=3 or views.shape[-1]!=3:
            raise ValueError('three observed RGB images are required (left -30, forward, right +30)')
        if plans.shape!=(3,24) or not np.isin(plans,[0,1,2]).all():
            raise ValueError('three 24-action candidates are required')
        words=self.encoder.words([target])
        ctx,cue=self.encoder.encode(views,words)
        text=F.normalize(words@self.encoder.proj,dim=-1)
        x=torch.as_tensor(self.old.route_inputs(ctx.astype('float32'),plans),device=self.device)[None]
        logits,fore=self.base(x,text)
        score=logits.sigmoid()
        if self.residual is not None:
            residual,_=self.residual(x,fore,text)
            score=score+self.cap*residual
        if first_observation:self.old.memory_start(episode_memory,cue[:,0])
        indicator=episode_memory.features(plans,xy,facing,time)
        weight=np.asarray([episode_memory.weights[k] for k in ('novelty','clearance','target_trace')])
        if len(self.encoder.cache)>900:self.encoder.cache.clear()
        return score[0].cpu().numpy()+indicator@weight,cue[:,0]
