"""Private-RGB evidence for revisiting a previously registered, trusted object."""
from dataclasses import dataclass
import math

import cv2
import numpy as np


# Geometric failures which abstain before scoring identity. Negative or
# uncertain identity outputs deliberately do not enter this secondary check.
MISSING_GEOMETRY = {
    "identity_missing_geometry", "identity_missing_two_visible_map_anchors",
    "identity_fewer_than_two_shared_timestamped_anchors", "identity_causal_rgb_window_incomplete",
}
CANVAS = 384
OBJECT = 96
LO = (CANVAS - OBJECT) // 2


@dataclass
class Region:
    gray: np.ndarray
    points: np.ndarray
    descriptors: object


def describe(rgb, box):
    """Normalize the object rectangle, retaining the surrounding physical landmarks."""
    b = np.asarray(box, dtype=float)
    if b.shape != (4,) or not np.isfinite(b).all():
        return None
    h,w = rgb.shape[:2]
    if min(b[:2]) <= 1 or b[2] >= w-1 or b[3] >= h-1 or min(b[2:]-b[:2]) < 12:
        return None
    sx,sy = OBJECT/(b[2:]-b[:2])
    matrix = np.array([[sx,0,LO-sx*b[0]],[0,sy,LO-sy*b[1]]])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    image = cv2.warpAffine(gray,matrix,(CANVAS,CANVAS),flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(np.full((h,w),255,np.uint8),matrix,(CANVAS,CANVAS),
                          flags=cv2.INTER_NEAREST)
    mask = cv2.erode(mask,np.ones((15,15),np.uint8))
    sift = cv2.SIFT_create(nfeatures=650,contrastThreshold=.014)
    kp,ds = sift.detectAndCompute(image,mask)
    return Region(image,np.asarray([k.pt for k in kp],dtype=np.float32).reshape(-1,2),ds)


def compare_regions(source, current):
    """Require interior structure and physical landmarks around the target."""
    if source is None or current is None or source.descriptors is None or current.descriptors is None:
        return None
    if min(len(source.points),len(current.points)) < 8:
        return None
    bf = cv2.BFMatcher()
    def ratios(left,right):
        return {m.queryIdx:m.trainIdx for m,n in bf.knnMatch(left,right,k=2)
                if m.distance < .82*n.distance}
    inside = lambda pts: np.all((pts >= LO+2) & (pts <= LO+OBJECT-2),axis=1)
    pairs = []
    for source_ids,current_ids in ((inside(source.points),inside(current.points)),
                                   (~inside(source.points),~inside(current.points))):
        aidx=np.flatnonzero(source_ids)
        bidx=np.flatnonzero(current_ids)
        if min(len(aidx),len(bidx))<2:
            continue
        aa,bb = source.descriptors[aidx],current.descriptors[bidx]
        forward,reverse = ratios(aa,bb),ratios(bb,aa)
        pairs.extend((aidx[a],bidx[b]) for a,b in forward.items() if reverse.get(b)==a)
    if len(pairs) < 6:
        return None
    a,b = zip(*pairs)
    pa,pb = source.points[list(a)],current.points[list(b)]
    # Multiple orientations at an identical SIFT keypoint count as one landmark.
    _,indices = np.unique(np.rint(np.column_stack((pa,pb))/3),axis=0,return_index=True)
    pa,pb = pa[indices],pb[indices]
    outside = lambda pts: np.any((pts < LO-8) | (pts > LO+OBJECT+8),axis=1)
    core = inside(pa) & inside(pb)
    ring = outside(pa) & outside(pb)
    # A nearby landmark need not be coplanar with the object. Fit only the
    # object, then test whether its neighborhood predicts the queried location.
    if core.sum() < 5:
        return None
    affine,_ = cv2.estimateAffine2D(pa[core],pb[core],method=cv2.RANSAC,ransacReprojThreshold=5)
    if affine is None:
        return None
    sv = np.linalg.svd(affine[:,:2],compute_uv=False)
    if np.linalg.det(affine[:,:2])<=0 or min(sv)<.5 or max(sv)>1.7:
        return None
    project = lambda points: points @ affine[:,:2].T + affine[:,2]
    center = np.array([CANVAS/2,CANVAS/2])
    delta = np.linalg.norm(project(center)-center)
    if delta>20:
        return None
    error = np.linalg.norm(project(pa)-pb,axis=1)
    inliers = core & (error<=5)
    outside_agree = ring & (error<=32)
    # Inliers concentrated on a single numeral or texture are insufficient.
    count = int(inliers.sum())
    spread = float(cv2.contourArea(cv2.convexHull(pb[inliers])) / OBJECT**2) if count>=3 else 0.
    if np.count_nonzero(outside_agree)>=2:
        offsets = pb[outside_agree]
        baseline = np.linalg.norm(np.ptp(offsets,axis=0))
    else:
        baseline = 0.
    warped = cv2.warpAffine(source.gray,affine,(CANVAS,CANVAS))
    inner = np.s_[LO+12:LO+OBJECT-12, LO+12:LO+OBJECT-12]
    aa=cv2.GaussianBlur(warped,(5,5),1.2)[inner].astype(np.float32)
    bb=cv2.GaussianBlur(current.gray,(5,5),1.2)[inner].astype(np.float32)
    aa-=aa.mean()
    bb-=bb.mean()
    similarity=float(np.sum(aa*bb)/max(1.,np.linalg.norm(aa)*np.linalg.norm(bb)))
    return count,spread,int(outside_agree.sum()),float(baseline),similarity


def strong_object(stats):
    return stats is not None and stats[0]>=8 and stats[1]>=.12 and stats[4]>=.80


def corroborated_surroundings(stats):
    return stats[2]>=2 and stats[3]>=OBJECT/2


class VisualWitnessMixin:
    def _init_report_memory(self):
        super()._init_report_memory()
        self._visual_history = {}
        self._visual_reference = {}
        self._rival_review_box = None

    def _rival_claims(self,node,box,obs,boxes):
        # When testing a competing owner, even one plausible match is enough
        # to withhold the report. A rival may first be queried on this frame.
        before = self._rival_review_box
        self._rival_review_box = tuple(box)
        try:
            return super()._rival_claims(node,box,obs,boxes)
        finally:
            self._rival_review_box = before

    def _support(self,fid,ref,obs,boxes):
        prior = super()._support(fid,ref,obs,boxes)
        key = (int(fid),tuple(ref.roi))
        if prior[0] is not None or prior[1] not in MISSING_GEOMETRY:
            if self._rival_review_box is None:
                self._visual_history.pop(key,None)
            return prior
        # TrustedReferenceMixin stores unaltered RGB + original registration
        # box for approved slots only. Do not promote route-only references.
        stored = self._trusted_pixels.get(key)
        if stored is None or not boxes:
            self._visual_history.pop(key,None)
            return prior
        if key not in self._visual_reference:
            self._visual_reference[key] = describe(*stored)
        from appearance import WORK_WIDTH, _rgb
        rgb = _rgb(obs.rgb)
        if rgb is None:
            self._visual_history.pop(key,None)
            return prior
        scale = rgb.shape[1]/WORK_WIDTH
        candidates = []
        for i,box in enumerate(boxes):
            if self._rival_review_box is not None and self._rival_review_box != tuple(box):
                continue
            region = describe(rgb,np.asarray(box)*scale)
            stats = compare_regions(self._visual_reference[key],region)
            if strong_object(stats):
                if self._rival_review_box is not None:
                    return box,"possible_visual_rival"
                candidates.append((i,corroborated_surroundings(stats)))
        if self._rival_review_box is not None:
            return prior
        if len(candidates)!=1:
            self._visual_history.pop(key,None)
            return prior
        idx,context = candidates[0]
        step = int(obs.step_count)
        signature = cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),(32,32),interpolation=cv2.INTER_AREA)
        previous = self._visual_history.get(key)
        accepted = False
        if previous is not None:
            last_step, last_frame, last_box, last_context, accepted_before = previous
            if last_step == step:
                return (boxes[idx],"multiview_visual_witness") if accepted_before and last_box == tuple(boxes[idx]) else prior
            if step == last_step+1:
                movement = cv2.absdiff(last_frame,signature).mean()
                old,now = np.asarray(last_box),np.asarray(boxes[idx])
                center_shift = np.linalg.norm((old[:2]+old[2:]-now[:2]-now[2:])/2)
                scale_change = abs(math.log(np.prod((old[2:]-old[:2])/(now[2:]-now[:2]))))
                accepted = (movement>=3 and center_shift<=WORK_WIDTH*.7 and
                            (center_shift>=3 or scale_change>=.1) and (context or last_context))
        self._visual_history[key] = (step,signature,tuple(boxes[idx]),context,accepted)
        return (boxes[idx],"multiview_visual_witness") if accepted else prior
