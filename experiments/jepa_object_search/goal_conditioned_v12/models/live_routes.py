"""Run the frozen first-choice head on current RGB, target, actions and episode memory."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from goal_ranker import GoalRanker, load_prior, previous


class LiveRoutes:
    def __init__(self, project, arm=None, seed=11):
        project = Path(project)
        d = project/'jepa_route_v12'/'models'
        self.old = previous(project)
        from encode import Encoder
        self.encoder = Encoder(project)
        self.device = self.encoder.device
        deployment = json.loads((d/'deployment.json').read_text())
        self.arm = deployment['selected'] if arm is None else arm
        cfg = deployment['arms'][self.arm]
        path = d/'weights'/f'{self.arm}_{seed}.pt'
        if self.old.sha(path)!=cfg['checkpoint_sha256'][str(seed)]:
            raise ValueError('selected route head digest mismatch')
        self.weight = cfg['head_weight']
        self.prior = (load_prior(self.old,project,seed,'jepa',self.device)
                      if self.weight < 1 else None)
        self.head = GoalRanker(self.encoder.proj.shape[1]).to(self.device)
        self.head.load_state_dict(torch.load(path,map_location=self.device,weights_only=True))
        self.head.eval()

    @torch.inference_mode()
    def scores(self, rgb, target, plans, episode_memory, time=2, xy=(0.,0.), facing=(1.,0.), first_observation=False):
        """RGB in yaw order -30,0,+30. Set first_observation=True once at the starting junction."""
        views = np.asarray(rgb)
        plans = np.asarray(plans)
        if views.ndim!=4 or views.shape[0]!=3 or views.shape[-1]!=3 or views.dtype!=np.uint8:
            raise ValueError('expected three uint8 HWC RGB views')
        if plans.shape != (3,24) or not np.isin(plans,[0,1,2]).all():
            raise ValueError('expected three equal 24-action HSSD plans')
        words = self.encoder.words([target])
        ctx,cue = self.encoder.encode(views,words)
        text = F.normalize(words@self.encoder.proj,dim=-1)
        x = torch.as_tensor(self.old.route_inputs(ctx.astype('float32'),plans),device=self.device)[None]
        own = self.head(x,text)[0].sigmoid()
        if self.prior is not None:
            own = (1-self.weight)*self.prior(x,text)[0].sigmoid() + self.weight*own
        scores = own[0].cpu().numpy()
        if first_observation:
            self.old.memory_start(episode_memory,cue[:,0])
        v = episode_memory.features(plans,xy,facing,time)
        coeff = [episode_memory.weights[k] for k in ('novelty','clearance','target_trace')]
        scores = scores + v@np.asarray(coeff)
        if len(self.encoder.cache)>900:
            self.encoder.cache.clear()
        return scores, cue[:,0]
