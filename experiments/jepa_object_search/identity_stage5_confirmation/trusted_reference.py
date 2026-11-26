"""Gate registered views using the original RGB and category proposals on that frame."""
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import cv2
import numpy as np


MIN_OCCUPANCY = 0.30
MIN_DETECTOR_SCORE = 0.70
# OpenCV's RNG is thread-local. Isolate the fixed GrabCut seed from the
# navigator's OpenCV operations, including robust pose estimation.
_SEGMENTATION = ThreadPoolExecutor(max_workers=1)


@dataclass(frozen=True)
class ReferenceEvidence:
    reason: str
    occupancy: float = 0.
    detection_score: float = 0.

    def eligible(self, minimum=MIN_OCCUPANCY):
        return bool(self.reason == "supported" and self.occupancy >= minimum)

    def rank(self):
        return self.occupancy * self.detection_score


def _foreground(rgb, detection):
    """Estimate a foreground mask inside a detector proposal with a local background ring."""
    height, width = rgb.shape[:2]
    x, y, u, v = detection
    pad = max(5, int(max(u-x, v-y) * .20))
    left, top = max(0,x-pad), max(0,y-pad)
    tile = rgb[top:min(height,v+pad), left:min(width,u+pad)]
    h, w = tile.shape[:2]
    scale = min(1., 192/max(w,h))
    size = (max(1, round(w*scale)), max(1, round(h*scale)))
    image = cv2.resize(tile, size, interpolation=cv2.INTER_AREA)
    sx, sy = size[0]/w, size[1]/h
    lo = np.floor([(x-left)*sx, (y-top)*sy]).astype(int)
    hi = np.ceil([(u-left)*sx, (v-top)*sy]).astype(int)
    if min(hi-lo)<4 or (hi-lo).prod() > .97*size[0]*size[1]:
        return None
    seed = np.zeros(image.shape[:2], dtype=np.uint8)
    seed[lo[1]:hi[1],lo[0]:hi[0]] = cv2.GC_PR_FGD
    if np.count_nonzero(seed==0)<5:
        return None
    cv2.setRNGSeed(0)
    cv2.grabCut(image, seed, None, np.zeros((1,65)), np.zeros((1,65)), 3, cv2.GC_INIT_WITH_MASK)
    mask = cv2.resize((seed==cv2.GC_PR_FGD).astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST)
    return mask[y-top:v-top,x-left:u-left]


def assess_reference(rgb, box, click, detections, *, min_score=MIN_DETECTOR_SCORE):
    """Return an observable foreground estimate, never an instance-label mask.

    Geometry and click are in the native RGB pixel grid. Match the registered
    click to a class proposal before estimating occupied pixels in the original
    registration rectangle. An unrelated object in a broad rectangle cannot
    contribute if its proposal excludes the click.
    """
    rgb = np.asarray(rgb)
    roi = np.asarray(box, dtype=float)
    point = np.asarray(click, dtype=float)
    if (rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8 or roi.shape!=(4,)
            or point.shape!=(2,) or not np.isfinite(np.r_[roi,point]).all()):
        return ReferenceEvidence("invalid_source")
    h,w = rgb.shape[:2]
    x,y = np.floor(roi[:2]).astype(int)
    u,v = np.ceil(roi[2:]).astype(int)
    if not (0<=x<=point[0]<u<=w and 0<=y<=point[1]<v<=h and min(u-x,v-y)>=6):
        return ReferenceEvidence("invalid_registration")
    best = ReferenceEvidence("no_class_box_at_click")
    for row in detections:
        det = np.asarray(row.get("bbox",row.get("xyxy")),dtype=float)
        score = float(row.get("score",0.))
        if det.shape!=(4,) or not np.isfinite(det).all() or not min_score<=score<=1:
            continue
        a,b = np.clip(np.floor(det[:2]),[0,0],[w,h]).astype(int)
        c,d = np.clip(np.ceil(det[2:]),[0,0],[w,h]).astype(int)
        if not (a<=point[0]<c and b<=point[1]<d and min(c-a,d-b)>=6):
            continue
        mask = _SEGMENTATION.submit(_foreground,rgb,(a,b,c,d)).result()
        if mask is None or not mask.any():
            if best.reason=="no_class_box_at_click":
                best=ReferenceEvidence("no_foreground")
            continue
        # Use only a connected component touching the registered click (with
        # a small tolerance for raster rounding and thin/ring-shaped objects).
        components, labels = cv2.connectedComponents(mask,connectivity=8)
        radius=max(2,min(6,round(.07*min(c-a,d-b))))
        px,py = int(point[0])-a,int(point[1])-b
        near=labels[max(0,py-radius):py+radius+1,max(0,px-radius):px+radius+1]
        if components<=1 or not near.any():
            if best.reason=="no_class_box_at_click":
                best=ReferenceEvidence("click_outside_foreground")
            continue
        votes=np.bincount(near.ravel(),minlength=components)
        label=1+int(np.argmax(votes[1:]))
        overlap=labels[max(0,y-b):min(d,v)-b,max(0,x-a):min(c,u)-a]
        fill=float(np.sum(overlap==label)/((u-x)*(v-y)))
        candidate=ReferenceEvidence("supported",fill,score)
        if candidate.rank()>best.rank():
            best=candidate
    return best


class TrustedReferenceMixin:
    def _init_report_memory(self):
        super()._init_report_memory()
        self._reference_evidence = {}
        self._confirmed_reference = {}
        self._trusted_pixels = {}
        self._supported_reference = None
        self._cleared_reference = None
        self._dispatched_reference = None
        self._routing_references = False

    def _source(self,fid,roi):
        key=(int(fid),tuple(roi))
        if key in self._trusted_pixels:
            return self._trusted_pixels[key]
        return super()._source(fid,roi)

    def _collect_views(self):
        super()._collect_views()
        # The original collector serves unreported items only. Recheck any
        # as-yet ungraded saved reference, including references owned by an
        # already reported item. Missing native RGB remains retryable.
        for oid,(fid,ref) in self._views.items():
            if oid in self._reference_evidence:
                continue
            record=self.instance_store.get_observation(oid)
            if record is None:
                continue
            raw=self._source(fid,ref.roi)
            if raw is None:
                continue
            from source_quality import coarse_category
            category=coarse_category(self._category(self._last_observation))
            if not category:
                continue
            rgb,box=raw
            norm=record.evidence.get("pixel_norm")
            click=np.asarray(norm,dtype=float)
            if click.shape==(2,):
                click=click*np.asarray(rgb.shape[1::-1])/1000.
            detections=self._boxes(("map",fid),rgb,category)
            if getattr(self,"_detector_failed",False):
                continue
            result=assess_reference(rgb,box,click,detections)
            self._reference_evidence[oid]=result
            if result.eligible():
                key=(int(fid),tuple(ref.roi))
                if key not in self._trusted_pixels:
                    stored=np.frombuffer(rgb.tobytes(),dtype=np.uint8).reshape(rgb.shape)
                    self._trusted_pixels[key]=(stored,tuple(box))
            self._note("reference_quality",None,observation=int(oid),frame=int(fid),
                       reason=result.reason,occupancy=result.occupancy,detector=result.detection_score,
                       trusted=result.eligible())

    def _references(self,node):
        # Both the recovery navigator and the verifier visit the last reference
        # first. Resolve membership through the current owner on every access.
        trusted=[(self._reference_evidence[i].rank(),self._views[i])
                 for i in node.observation_ids if i in self._views and i in self._reference_evidence
                 and self._reference_evidence[i].eligible()]
        pinned=self._confirmed_reference.get(node.iid)
        if pinned is not None:
            # An accepted historical view survives source-list compaction.
            # New references can be tried, but cannot displace the pinned one
            # as the primary comparison for an already acknowledged instance.
            trusted=[item for item in trusted if item[1][1] is not pinned[1]]
        ordered=[pair for _,pair in sorted(trusted,key=lambda item:item[0])]
        if not ordered and pinned is None and self._routing_references and not node.reported:
            return super()._references(node)
        return ordered+([pinned] if pinned is not None else [])

    @contextmanager
    def _reference_purpose(self,routing):
        prior=self._routing_references
        self._routing_references=routing
        try:
            yield
        finally:
            self._routing_references=prior

    def _auto_report(self,obs):
        # The inherited function also schedules nearby instances for a new view.
        # Nested report checks below always re-enter the trusted-only context.
        with self._reference_purpose(True):
            return super()._auto_report(obs)

    def _recover_report(self,node,obs):
        with self._reference_purpose(True):
            return super()._recover_report(node,obs)

    def _begin_confirm_view(self,node,obs,pose,scale):
        with self._reference_purpose(True):
            return super()._begin_confirm_view(node,obs,pose,scale)

    def _report_gate(self,node,obs):
        with self._reference_purpose(False):
            return super()._report_gate(node,obs)

    def _support(self,fid,ref,observation,boxes):
        box,reason=super()._support(fid,ref,observation,boxes)
        self._supported_reference=(int(observation.step_count),int(fid),ref) if box is not None else None
        return box,reason

    def _rival_claims(self,node,box,obs,boxes):
        # Checking rivals may itself invoke _support. Capture the reference for
        # this candidate before inspecting any rival.
        candidate=self._supported_reference
        disputed=super()._rival_claims(node,box,obs,boxes)
        cleared=(not disputed and candidate is not None and candidate[0]==int(obs.step_count)
                 and any(fid==candidate[1] and ref is candidate[2] for fid,ref in self._references(node)))
        self._cleared_reference=(node.iid,candidate) if cleared else None
        return disputed or not cleared

    def _stage_report(self,node,observation,reason):
        cleared=self._cleared_reference
        if cleared is None or cleared[0]!=node.iid or cleared[1][0]!=int(observation.step_count):
            raise RuntimeError("report missing a trusted, verified, conflict-cleared reference")
        action=super()._stage_report(node,observation,reason)
        self._dispatched_reference=cleared
        return action

    def _settle_report(self,observation):
        pending=self._pending_report
        previous_count=self._reported_count
        super()._settle_report(observation)
        if pending is None or self._pending_report is not None:
            return
        dispatched=self._dispatched_reference
        self._dispatched_reference=None
        if (self._reported_count!=previous_count+1 or dispatched is None or
                dispatched[0]!=pending.instance or dispatched[1][0]!=pending.step):
            return
        owners={n.iid:n for oid in pending.observations
                if (n:=self.instance_store.instance_for_observation(oid)) is not None}
        if len(owners)!=1:
            return
        node=next(iter(owners.values()))
        if node.reported and node.iid not in self._confirmed_reference:
            _,fid,ref=dispatched[1]
            self._confirmed_reference[node.iid]=(fid,ref)
            self._note("confirmed_reference_pinned",node.iid,frame=fid)
