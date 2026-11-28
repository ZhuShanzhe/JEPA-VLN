"""Re-select a local same-instance view from the actual pose after contact or an unconfirmed arrival."""
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
VIEWS = HERE.parent / "new_view_20261129"
sys.path.insert(0, str(VIEWS))
from observation_actor import LEGACY, box_anchor, assess_plans
from long_data import FrozenImageEncoder
from mixed_checker import MixedPathChecker
from prior_cues_score import visual_input
from view_geometry import features, map_features, nominal, options, turns, SOURCE_H
from view_model import ViewPredictor


def axes(yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return np.asarray([[c,s],[-s,c]])  # row vector local (right,forward) -> initial frame


def position(frame):
    return np.asarray([frame["x"],frame["z"]],float)


def keyframes(history):
    """Keep genuinely different acquired frames. Repeated images after contact add no new geometry."""
    chosen=[]
    for frame in reversed(history):
        if any(np.linalg.norm(position(frame)-position(ref))<.12 and
               math.cos(frame["yaw"]-ref["yaw"])>.985 for ref in chosen):
            continue
        chosen.append(frame)
        if len(chosen)==8:break
    return chosen


def local_depth_map(history):
    """Fuse the recorded depth surfaces into the auxiliary +30 degree projection used in training."""
    current=history[-1]
    cpos=position(current)
    target=axes(current["yaw"]+math.pi/6)
    result=np.full((192,192),np.inf,np.float32)
    v,u=np.mgrid[:192:2,:192:2]
    for frame in keyframes(history):
        dep=np.asarray(frame["depth"],np.float32).squeeze()
        if dep.shape!=(192,192):raise ValueError("expected native 192x192 depth")
        z=dep[::2,::2]
        valid=np.isfinite(z)&(z>.05)&(z<15)
        zv=z[valid]
        loc=np.column_stack(((u[valid]+.5-96)*zv/96,zv))
        point=(loc@axes(frame["yaw"])+position(frame)-cpos)@target.T
        zz=point[:,1]
        ok=zz>.05
        xp=np.floor(96+96*point[ok,0]/zz[ok]).astype(int)
        yp=np.floor(96+(v[valid][ok]+.5-96)*zv[ok]/zz[ok]).astype(int)
        for dy,dx in ((0,0),(0,1),(1,0),(1,1)):
            xx=xp+dx;yy=yp+dy
            inside=(xx>=0)&(xx<192)&(yy>=0)&(yy<192)
            np.minimum.at(result,(yy[inside],xx[inside]),zz[ok][inside].astype(np.float32))
    result[~np.isfinite(result)]=0
    return result


def contact_edges(history):
    """A failed forward action rules out retrying that approach, not an entire region."""
    edges=[]
    for before,after in zip(history,history[1:]):
        if not after.get("bump",False):continue
        direction=axes(before["yaw"])[1]
        edges.append((position(before)+.25*direction,direction))
    return edges


def routes(anchor, remaining, history):
    current=history[-1]
    origin=position(current);rot=axes(current["yaw"])
    visited=np.array([position(x) for x in history])
    edges=contact_edges(history)
    for n in (2,4,6,8,10,12):
        for bearing in (0,-1,1,-2,2,-3,3,-4,4):
            ang=bearing*math.pi/6
            x,z=-.25*n*math.sin(ang),.25*n*math.cos(ang)
            end=np.array([x,z])@rot+origin
            if np.linalg.norm(visited-end,axis=1).min()<.21:continue
            # Aim the terminal camera at the remembered instance, including lateral approaches.
            facing=round(math.atan2(x-anchor[0],anchor[2]-z)/(math.pi/6))
            members=[]
            for body in options(n)[:3]:
                plan=turns(bearing)+body+turns(facing-bearing)
                if len(plan)>remaining:continue
                states=nominal(plan);fwd=states[states[:,4]>.5]
                pts=fwd[:,:2]@rot+origin
                dirs=np.column_stack((-fwd[:,2],fwd[:,3]))@rot
                if any(np.any((np.linalg.norm(pts-point,axis=1)<.215)&(dirs@direction>.40))
                       for point,direction in edges):
                    continue
                members.append(plan)
            if members:yield (n,bearing),members


def survey(history,remaining):
    if remaining<2:
        return dict(actions=[],reason="no_translation_budget")
    current=history[-1]
    nearby=[x for x in history if np.linalg.norm(position(x)-position(current))<.25]
    tested=[]
    for relative in (-2,2,-1,1,-3,3,-4,4):
        motion=turns(relative)
        if len(motion)>=remaining:continue
        heading=current["yaw"]+relative*math.pi/6
        novelty=min(math.acos(np.clip(math.cos(heading-f["yaw"]),-1,1)) for f in nearby)
        if novelty<.45:continue
        tested.append((abs(abs(relative)-2),-novelty,len(motion),relative,motion))
    if tested:
        return dict(actions=list(min(tested)[-1]),reason="survey_unseen_side")
    return dict(actions=[],reason="local_options_exhausted")


class RoutePlanner:
    def __init__(self,checker,view,encoder):
        self.checker=checker
        self.view=view
        self.encoder=encoder

    @classmethod
    def load(cls, device=None):
        return cls(MixedPathChecker.load(LEGACY/"private_checker_mixed/checker.joblib"),
                   ViewPredictor.load(VIEWS/"private_model/jepa.pt"),FrozenImageEncoder(HERE.parent))

    def choose(self,source_context,source_box,source_depth,history,remaining,*,comparison=False):
        """All poses are actual odometric poses in the initial right/forward, positive-left frame.

        Return a single budgeted action sequence. A subsequent contact cancels outstanding actions.
        'survey_unseen_side' is an ordinary paid turn and requires a fresh depth frame before replanning.
        """
        if not history or remaining<1:return dict(actions=[],reason="empty_history_or_budget")
        initial=box_anchor(source_box,np.asarray(source_depth).squeeze())
        if initial is None:return dict(actions=[],reason="missing_source_anchor")
        goal,box=initial
        current=history[-1]
        goal_local=(goal[[0,2]]-position(current))@axes(current["yaw"]).T
        anchor=np.array([goal_local[0],goal[1],goal_local[1]])
        depth=np.asarray(current["depth"]).squeeze()
        if depth.shape!=(192,192) or not (np.isfinite(depth)&(depth>.05)).any():
            return dict(actions=[],reason="missing_current_depth")
        # Preserve the initial angular object size and depth as a metric-size cue.
        # Projected coordinates and optical depth at each proposed view use the updated anchor.
        variants=list(routes(anchor,remaining,history))
        if not variants:return survey(history,remaining)
        plans=[plan for _,family in variants for plan in family]
        ok,pr=assess_plans(plans,depth,self.checker)
        allowed=[];start=0
        for view,variants_for_view in variants:
            indices=[i for i in range(start,start+len(variants_for_view)) if ok[i]]
            if indices:
                i=min(indices,key=lambda i:len(plans[i]))
                allowed.append((view,i))
            start+=len(variants_for_view)
        if not allowed:return survey(history,remaining)
        measured_map=local_depth_map(history)
        context=np.asarray(source_context,np.float32).reshape(2048).copy()
        if current.get("rgb") is not None:
            current_rgb=np.asarray(current["rgb"],np.uint8)[...,:3]
            context[:1024]=self.encoder.encode([visual_input(current_rgb)])[0]
        idx=np.array([i for _,i in allowed])
        pose=np.stack([features(plans[i],SOURCE_H,0.,anchor,box,depth)[0] for i in idx])
        memory=np.stack([map_features(plans[i],anchor,measured_map) for i in idx])
        pv=self.view.predict(np.repeat(context[None],len(idx),axis=0),pose,memory)["p_visible"]
        score=pv*pr[idx]
        best=min(range(len(idx)),key=lambda k:(-float(score[k]),len(plans[idx[k]]),k))
        i=idx[best];n,bearing=allowed[best][0]
        result=dict(actions=list(plans[i]),reason="reactive_jepa_view",local_distance_m=n*.25,
                    relative_bearing_deg=bearing*30,p_reach=float(pr[i]),p_visible=float(pv[best]),
                    candidate_views=len(variants),admitted_views=len(allowed),map_keyframes=len(keyframes(history)),
                    contact_edges=len(contact_edges(history)))
        if comparison:
            from view_model import current_inputs
            _,prior=current_inputs(np.repeat(context[None,:],len(idx),axis=0),pose,memory)
            geom=pr[idx]/(1+np.exp(-prior))
            highest=min(range(len(idx)),key=lambda k:(-float(geom[k]),len(plans[idx[k]]),k))
            nearest=min(range(len(idx)),key=lambda k:(allowed[k][0][0],len(plans[idx[k]]),k))
            result["control_actions"]={"projected_area":list(plans[idx[highest]]),
                                       "nearest":list(plans[idx[nearest]])}
        return result
