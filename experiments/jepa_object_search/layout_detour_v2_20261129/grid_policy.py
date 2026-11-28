"""Heading-aware shortest search of observed free floor and short, rechecked physical actions."""
import heapq,json,math,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
V1=HERE.parent/'layout_detour_20261129'
REACTIVE=HERE.parent/'reactive_view_20261129'
sys.path[:0]=[str(V1),str(REACTIVE)]
from layout_model import LayoutMap,encode_current,describe
from layout_paths import axes,path,turns
from layout_policy import LayoutPlanner
from observation_actor import assess_plans,box_anchor
from reactive_route import position,survey
from prior_cues_score import visual_input

def known_routes(patch,xy,yaw,anchor,budget):
    """Dijkstra search in discretised x,z and heading. Physical positions stay continuous."""
    angles=yaw+np.arange(12)*np.pi/6
    direction=np.stack([-np.sin(angles),np.cos(angles)],1)
    horizon=min(budget,17)
    # Scalar lookup in the observed patch; all predicted intermediate points must be free.
    cell=patch.cell_m;size=len(patch.cells);radius=size//2;array=patch.cells
    def free(x,z):
        i=round((x-patch.cx)/cell)+radius;j=round((z-patch.cz)/cell)+radius
        return 0<=i<size and 0<=j<size and array[j,i]==1
    heap=[(0,0,xy[0],xy[1],0,(),0)];costs={};seen=0;endpoints={}
    while heap and seen<2300:
        cost,_,x,z,h,moves,fwd=heapq.heappop(heap)
        key=(round((x-xy[0])/cell),round((z-xy[1])/cell),h)
        if key in costs:continue
        costs[key]=cost;seen+=1
        if cost>=horizon:continue
        for offset in (0,-1,1,-2,2,-3,3,-4,4):
            nh=(h+offset)%12
            nx=x+direction[nh,0]*.25;nz=z+direction[nh,1]*.25
            extra=turns(offset)+(1,);nc=cost+len(extra)
            if nc>horizon or not free((nx+x)*.5,(nz+z)*.5) or not free(nx,nz):continue
            nkey=(round((nx-xy[0])/cell),round((nz-xy[1])/cell),nh)
            if nkey in costs:continue
            plan=moves+extra
            # Each node is a physically distinct reachable position. Face the cue before inspection.
            gaze=round((math.atan2(nx-anchor[0],anchor[1]-nz)-yaw)/(math.pi/6))%12
            terminal=plan+turns(gaze-nh)
            k=(round((nx-xy[0])/.25),round((nz-xy[1])/.25),gaze)
            if len(terminal)<=budget and (k not in endpoints or len(terminal)<len(endpoints[k])):
                endpoints[k]=terminal
            heapq.heappush(heap,(nc,len(costs)*20+nh,nx,nz,nh,plan,fwd+1))
    return list(dict.fromkeys(endpoints.values()))


def prefix(plan):
    """At most two real translations per call; include any final observation-facing turns."""
    n=0
    for i,a in enumerate(plan):
        if a==1:
            n+=1
            if n==3:return plan[:i]
    return plan


def traverses_bend(plan):
    h=0;directions=set()
    for a in plan:
        if a==1:directions.add(h%12)
        else:h+=1 if a==2 else -1
    return len(directions)>=2


class GridPlanner(LayoutPlanner):
    def __init__(self,previous):
        super().__init__(previous)
        self.settings=json.loads((HERE/'calibration.json').read_text())['chosen']
        self.context_key=None;self.last_t=-1;self.goal=None;self.suffix=()

    @torch.inference_mode()
    def choose(self,source_context,source_box,source_depth,history,remaining,*,external=None):
        current=history[-1];xy=position(current);yaw=current['yaw']
        key=(hash(np.asarray(source_depth).tobytes()),tuple(source_box),current.get('episode'))
        t=len(history)-1
        if key!=self.context_key or t<=self.last_t or any(f.get('bump',False) for f in history[self.last_t+1:]):
            self.goal=None;self.suffix=()
        self.last_t=t;self.context_key=key
        source=box_anchor(source_box,np.asarray(source_depth).squeeze())
        if source is None or remaining<1:return dict(actions=[],reason='no_anchor_or_budget')
        patch=LayoutMap.from_depth_history(history,episode=current.get('episode','local-inspection'),floor=current.get('floor',0),
                    frame_id=current.get('frame_id',t),external=external if external is not None else current.get('layout_patch'))
        def observed(p):
            pts=path(p)@axes(yaw)+xy
            if not len(pts):return True
            prev=np.vstack([xy,pts[:-1]])
            pts=np.vstack([pts,(prev+pts)*.5])
            return bool(np.all(patch.query(pts[:,0],pts[:,1])==1))
        if self.suffix and len(self.suffix)<=remaining and observed(self.suffix):
            step=prefix(self.suffix)
            if 1 not in step or assess_plans([step],current['depth'],self.previous.checker)[0][0]:
                self.suffix=self.suffix[len(step):]
                return dict(actions=list(step),reason='resume_grid_waypoint',mode='committed',map_keyframes=t+1)
        self.suffix=();self.goal=None
        goal,box=source
        endpoints=known_routes(patch,xy,yaw,goal[[0,2]],remaining)
        prior=self.previous.choose(source_context,source_box,source_depth,history,remaining)
        op=tuple(prior['actions'])
        if 1 in op and observed(op):endpoints.append(op)
        visited=np.stack([position(f) for f in history])
        plans=[]
        for p in dict.fromkeys(endpoints):
            end=path(p)[-1]@axes(yaw)+xy
            if np.min(np.linalg.norm(visited-end,axis=1))>=.22:plans.append(p)
        if not plans:
            return survey(history,remaining)
        loc=(goal[[0,2]]-xy)@axes(yaw).T;anchor=np.array([loc[0],goal[1],loc[1]])
        ctx=np.array(source_context,np.float32).copy().ravel()
        ctx[:1024]=self.previous.encoder.encode([visual_input(current['rgb'])])[0]
        cur=encode_current(ctx,anchor,box)
        feats=np.stack([describe(p,patch,xy,yaw,anchor,box) for p in plans])
        inp=np.column_stack((np.repeat(cur[None],len(plans),axis=0),feats))
        _,logits=self.net(((torch.from_numpy(inp.astype(np.float32))-self.mu)/self.sd).clamp(-6,6))
        pr=torch.sigmoid(logits).numpy()
        scores=pr[:,1]*pr[:,0]**self.settings['alpha']-self.settings['cost_penalty']*feats[:,-2]
        ids=sorted(range(len(plans)),key=lambda i:(-scores[i],len(plans[i])))[:32]
        early=[prefix(plans[i]) for i in ids]
        ok,pcheck=assess_plans(early,current['depth'],self.previous.checker)
        allowed=[j for j,o in enumerate(ok) if o]
        if not allowed:return survey(history,remaining)
        j=max(allowed,key=lambda j:(scores[ids[j]]*pcheck[j],-len(plans[ids[j]])))
        idx=ids[j];chosen=plans[idx];step=early[j]
        self.suffix=chosen[len(step):]
        self.goal=path(chosen)[-1]@axes(yaw)+xy
        return dict(actions=list(step),reason='observed_grid_waypoint',mode='curved' if traverses_bend(chosen) else 'straight',
                   range_m=float(np.linalg.norm(self.goal-xy)),target=list(map(float,self.goal)),score=float(scores[idx]),
                   p_reach=float(pcheck[j]),p_visible=float(pr[idx,1]),n_candidates=len(plans),n_admitted=len(allowed),
                   map_keyframes=len(history))
