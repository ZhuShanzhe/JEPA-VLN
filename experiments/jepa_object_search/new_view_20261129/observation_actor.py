"""A bounded local action for a known visual instance, after one leftward scan and heading restore."""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

from view_dispatch import fallback
from view_geometry import S, SOURCE_H, endpoints, features, map_features, options
from view_model import ViewPredictor

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "path_feedback_20261128"
OLD = HERE.parent / "pose_horizon_20261128"
sys.path[:0] = [str(LEGACY), str(OLD)]
from local_repair import select_or_repair
from mixed_checker import MixedPathChecker
from observation_head import HorizonCameraPredictor
from path_geometry import from_depth, path_features
from replan import RepairScorer


def box_anchor(box, depth):
    """Use the supplied detector rectangle and the median central depth in metres."""
    l,t,r,b = map(int, box)
    if not 0 <= l < r <= S or not 0 <= t < b <= S:
        raise ValueError("invalid xyxy detector box for native image")
    u,v = (l+r)/2, (t+b)/2
    patch = depth[max(0,int(v)-2):int(v)+3, max(0,int(u)-2):int(u)+3]
    valid = patch[np.isfinite(patch) & (patch>.05)]
    if not len(valid):
        return None
    z = float(np.median(valid))
    return np.array([(u-S/2)*z/(S/2), (S/2-v)*z/(S/2), z]), [u/S,v/S,(r-l)/S,(b-t)/S,min(z,10.)]


def assess_plans(plans, depth, checker):
    width = max(map(len, plans))
    a = np.zeros((len(plans),width),np.int8)
    x = np.zeros((len(plans),width,17),np.float32)
    frame = from_depth(depth)
    for i,p in enumerate(plans):
        a[i,:len(p)] = p
        x[i,:len(p)] = path_features(p,frame)
    v = checker.predict(a,x)
    return v["decision"]=="accept", v["p_reach"]


def more_evidence(reason):
    return dict(status="request_observation", actions=[], reason=reason)


class ObservationActor:
    def __init__(self, checker, old_predictor, repair, view_predictor):
        self.checker = checker
        self.old = old_predictor
        self.repair = repair
        self.view = view_predictor

    @classmethod
    def load(cls, device="cpu"):
        return cls(MixedPathChecker.load(LEGACY/"private_checker_mixed/checker.joblib"),
                   HorizonCameraPredictor.from_checkpoints(OLD/"private_train",device),
                   RepairScorer(LEGACY/"private_repair/with_hint.pt"),
                   ViewPredictor.load(HERE/"private_model/jepa.pt"))

    def choose(self, context, box, depth, left_depth, old_forward_steps, remaining_actions):
        """context is the 2048-D frozen full image + box encoding (same instance throughout).

        remaining_actions is the balance after paying for the scan; left_depth was captured
        at the same translation, with +30 degree yaw, then the agent returned to the source yaw.
        The supported request is a straight 1, 2 or 3 m observation waypoint from the source.
        Returns executable primitive actions (1 forward 0.25 m, 2/3 left/right 30 degrees).
        """
        if old_forward_steps not in (4,8,12):
            raise ValueError("unvalidated original waypoint")
        depth=np.asarray(depth).squeeze()
        if depth.shape!=(S,S): raise ValueError("depth must be native 192x192 metres")
        if remaining_actions < old_forward_steps: return more_evidence("insufficient_remaining_budget")
        source=box_anchor(box,depth)
        if source is None: return more_evidence("missing_source_depth")
        anchor,info=source
        ctx=np.asarray(context,np.float32).reshape(2048)
        if not np.isfinite(ctx).all(): raise ValueError("invalid visual embedding")
        @lru_cache(maxsize=None)
        def pose(plan):
            return features(plan,SOURCE_H,0.,anchor,info,depth)[0]
        old_plans=options(old_forward_steps)
        original=select_or_repair(old_plans[:3],depth,ctx,pose,self.checker,self.old,
                                  self.repair,remaining_actions)
        if original["status"]=="execute":
            return dict(status="execute",actions=original["actions"],reason="original_waypoint",view_id=-1)
        affordable=[p for p in old_plans if len(p)<=remaining_actions]
        old_ok,_=assess_plans(affordable,depth,self.checker)
        if old_ok.any():
            return more_evidence("original_waypoint_not_exhausted")
        if left_depth is None:
            return more_evidence("missing_local_map")
        left=np.asarray(left_depth).squeeze()
        if left.shape!=(S,S): raise ValueError("expected calibrated left scan")
        if not (np.isfinite(left)&(left>.05)).any():
            return more_evidence("missing_local_map")
        candidates=[(v,i,p) for k,v,variants in endpoints(old_forward_steps,anchor) if k=="alt"
                    for i,p in variants if len(p)<=remaining_actions]
        if not candidates: return more_evidence("no_budgeted_alternative")
        ids,pids,plans=zip(*candidates)
        accepted,prob=assess_plans(plans,depth,self.checker)
        if not accepted.any(): return more_evidence("no_admitted_alternative")
        pred=self.view.predict(np.repeat(ctx[None,:],len(plans),axis=0),np.asarray([pose(p) for p in plans]),
                               np.asarray([map_features(p,anchor,left) for p in plans]))["p_visible"]
        chosen=fallback(None,np.full(len(plans),"alt"),np.asarray(ids),accepted,
                        np.asarray([len(p) for p in plans]),remaining_actions,prob,pred)
        return dict(status="execute",reason="new_observation_waypoint",actions=list(plans[chosen]),
                    view_id=int(ids[chosen]),plan_id=int(pids[chosen]),p_reach=float(prob[chosen]),
                    p_visible_if_reached=float(pred[chosen]))

    def from_images(self, encoder, rgb, box, depth, left_depth, old_forward_steps, remaining_actions):
        image=np.asarray(rgb)
        if image.shape!=(S,S,3) or image.dtype!=np.uint8: raise ValueError("expected native RGB uint8")
        l,t,r,b=map(int,box)
        if not 0<=l<r<=S or not 0<=t<b<=S: raise ValueError("invalid image ROI")
        from prior_cues_score import visual_input
        context=encoder.encode([visual_input(image),visual_input(image[t:b,l:r])]).ravel()
        return self.choose(context,box,depth,left_depth,old_forward_steps,remaining_actions)
