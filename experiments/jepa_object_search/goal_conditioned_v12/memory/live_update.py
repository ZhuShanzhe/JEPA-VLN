"""Use a real outbound and return trace to rescore the uninspected candidates."""
import json
from pathlib import Path
import sys

import numpy as np
from replan import evidence_features, second_scores


class ObservedRouteUpdate:
    def __init__(self, project):
        project=Path(project)
        sys.path.insert(0,str(project/'jepa_hidden_v11/models'))
        from encode import Encoder
        from physical_return import advance
        self.encoder=Encoder(project)
        self.advance=advance
        self.head=json.loads((project/'jepa_route_v12/memory/trained_replan.json').read_text())

    def select(self, *, memory, target, start_rgb, plans, first, scores_at_start, bonus_at_start,
               outbound_rgb, outbound_collisions, return_actions, return_collisions, return_rgb,
               start_xy, start_yaw, start_time, returned):
        """Update caller-owned memory. An invalid return yields no second choice."""
        ctx=np.asarray(start_rgb); out=np.asarray(outbound_rgb); back=np.asarray(return_rgb)
        plans=np.asarray(plans)
        home=np.asarray(return_actions)
        if ctx.shape[0]!=3 or out.shape[0]!=4 or back.shape[0]!=6 or plans.shape!=(3,24) or home.shape!=(36,):
            raise ValueError('expected three starting views, four outbound views and six return views at six-action intervals')
        if not 0<=first<3 or len(outbound_collisions)!=24 or len(return_collisions)!=36:
            raise ValueError('invalid selected route or collision trace length')
        tok=self.encoder.words([target])
        import torch.nn.functional as F
        text=F.normalize(tok@self.encoder.proj,dim=-1).cpu().numpy()[0]
        enc=self.encoder
        ctx_feat,ctx_cue=enc.encode(ctx,tok)
        seen,seen_cue=enc.encode(out,tok)
        _,back_cue=enc.encode(back,tok)
        if len(enc.cache)>450:
            enc.cache.clear()
        xy,yaw,t=self.advance(memory,np.asarray(start_xy,float),start_yaw,start_time,
                              plans[first],outbound_collisions,dict(zip(range(6,25,6),seen_cue[:,0])))
        xy,yaw,t=self.advance(memory,xy,yaw,t,home,return_collisions,dict(zip(range(6,37,6),back_cue[:,0])))
        if not returned:
            return None
        features_now=memory.features(plans,xy,[np.cos(yaw),np.sin(yaw)],t)
        weights=np.array([memory.weights[k] for k in ('novelty','clearance','target_trace')])
        features=evidence_features(ctx_feat.astype(float),text,plans,first,seen.astype(float),seen_cue[:,0],
                                   ctx_cue[:,0],outbound_collisions,features_now)
        ranked=second_scores(np.asarray(scores_at_start),np.asarray(bonus_at_start),features_now@weights,
                             features,self.head,first)
        return int(np.argmax(ranked))
