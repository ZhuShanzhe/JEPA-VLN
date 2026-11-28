"""Layout-aware recovery for ReactiveInspection, using the same primitive actions, budget and checker."""
import math
from pathlib import Path
import numpy as np
import torch
from layout_model import HERE,LayoutMap,encode_current,describe,learned_net
from layout_paths import axes,path,proposals
from observation_actor import assess_plans,box_anchor
from reactive_route import position,contact_edges
from prior_cues_score import visual_input


class LayoutPlanner:
    def __init__(self,previous):
        data=torch.load(HERE/'private_layout.pt',weights_only=True,map_location='cpu')
        self.mu,self.sd=data['mu'],data['sd']
        self.net=learned_net(len(self.mu))
        self.net.load_state_dict(data['weights'])
        self.net.eval()
        self.previous=previous

    @torch.inference_mode()
    def choose(self,source_context,source_box,source_depth,history,remaining,*,external=None):
        current=history[-1]
        prior=self.previous.choose(source_context,source_box,source_depth,history,remaining)
        initial=box_anchor(source_box,np.asarray(source_depth).squeeze())
        if initial is None or remaining<6:return prior
        goal,box=initial
        xy=position(current);yaw=current['yaw']
        local=(goal[[0,2]]-xy)@axes(yaw).T
        anchor=np.array([local[0],goal[1],local[1]])
        ctx=np.asarray(source_context,np.float32).copy().ravel()
        ctx[:1024]=self.previous.encoder.encode([visual_input(current['rgb'])])[0]
        source=encode_current(ctx,anchor,box)
        episode=current.get('episode','local-inspection')
        patch=LayoutMap.from_depth_history(history,episode=episode,floor=current.get('floor',0),
            frame_id=current.get('frame_id',len(history)-1),external=external if external is not None else current.get('layout_patch'))
        candidates=proposals(anchor,remaining)
        old=tuple(prior['actions'])
        old_has_translation=1 in old
        if old_has_translation:candidates.append((old,0,0))
        actions=[];fields=[];bends=[];visited=np.stack([position(f) for f in history])
        edge_list=contact_edges(history)
        for p,h,b in candidates:
            points=path(p)@axes(yaw)+xy
            delta=points-np.vstack([xy,points[:-1]])
            if any(np.any((np.linalg.norm(points-ep,axis=1)<.215)&(delta@ed>.10)) for ep,ed in edge_list):continue
            if np.min(np.linalg.norm(visited-points[-1],axis=1))<.23:continue
            f=describe(p,patch,xy,yaw,anchor,box)
            # Filled body-clearance walls are hard constraints for the proposed route.
            if f[-7]>0:continue
            actions.append(p);fields.append(f);bends.append(b)
        if not actions:return prior
        mat=np.column_stack([np.repeat(source[None],len(fields),axis=0),np.stack(fields)])
        _,logits=self.net(((torch.from_numpy(mat.astype(np.float32))-self.mu)/self.sd).clamp(-6,6))
        prob=torch.sigmoid(logits).numpy()
        scores=prob.prod(1)
        # A learned-shape curve must improve on an available original route, using the same score.
        baseline_score=max((scores[i] for i,p in enumerate(actions) if p==old),default=0.)
        order=sorted((i for i in range(len(actions)) if bends[i] and scores[i]>=baseline_score+.035),
                     key=lambda i:(-scores[i],len(actions[i]),i))[:36]
        if not order:return prior
        valid,pcheck=assess_plans([actions[i] for i in order],current['depth'],self.previous.checker)
        allowed=[j for j in range(len(order)) if valid[j]]
        if not allowed:return prior
        # For near-identical viewpoints, retain the shortest checked route.
        by_view={}
        for j in allowed:
            idx=order[j];p=actions[idx]
            heading=(p.count(2)-p.count(3))%12
            endpoint=tuple(np.rint(path(p)[-1]/.125).astype(int))+(heading,)
            if endpoint not in by_view or len(p)<len(actions[order[by_view[endpoint]]]):by_view[endpoint]=j
        j=max(by_view.values(),key=lambda j:(float(scores[order[j]])*float(pcheck[j]),-len(actions[order[j]])))
        i=order[j]
        return dict(actions=list(actions[i]),reason='layout_jepa_detour',mode='curved',score=float(scores[i]),
                    p_visible=float(prob[i,1]),p_reach=float(pcheck[j]),relative_bearing_deg=int(bends[i])*30,
                    exposure=float(fields[i][-8]),candidate_views=len(candidates),admitted_views=len(by_view),
                    map_keyframes=len(history),contact_edges=len(edge_list))
