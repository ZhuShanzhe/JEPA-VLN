"""Optional cross-view verifier at the existing MOST-Nav report gate."""
import os
import hashlib
from pathlib import Path

import numpy as np

from appearance import _rgb, WORK_WIDTH
from bridge import Worker
from map_views import metric_view, rgb_aligned
from provider_agent import BoxComparisonAgent


def mapped_frame(snapshot, native, rgb, boxes, scale, stride=8):
    """Project sparse map depth into the RGB pixel grid. Never count interpolated points."""
    if snapshot is None or scale is None or native is None or not rgb_aligned(snapshot,native,stride):
        return None
    import cv2
    hn,wn=native.shape[:2]
    h,w=rgb.shape[:2]
    if wn*h!=w*hn or not rgb_aligned(snapshot,rgb,stride):
        return None
    view=metric_view(snapshot,scale,stride)
    if view is None:
        return None
    grid=view["grid"]
    pose=view["c2w"]
    z=(grid-pose[:3,3])@pose[:3,2]
    yy,xx=np.indices(z.shape)
    yy,xx=yy*stride,xx*stride
    keep=(yy<hn)&(xx<wn)&np.isfinite(z)&(z>.01)&(z<100)
    depth=np.full((hn,wn),np.nan,dtype=np.float32)
    depth[yy[keep],xx[keep]]=z[keep]
    # Use the native pixel grid for the model, consistent with intrinsics/pixel support.
    scaled=cv2.resize(rgb,(wn,hn),interpolation=cv2.INTER_AREA) if (hn,wn)!=(h,w) else rgb
    native_boxes=np.asarray(boxes,dtype=float).reshape(-1,4)*[wn/w,hn/h,wn/w,hn/h]
    return scaled,native_boxes,depth,view["K"],pose


class IdentityMemoryAgent(BoxComparisonAgent):
    def _init_report_memory(self):
        super()._init_report_memory()
        self._identity_mode=os.environ.get("NAV_STAGE3_IDENTITY","original")
        if self._identity_mode not in ("original","direct","jepa"):
            raise ValueError("unknown identity mode")
        old=getattr(self,"_identity_worker",None)
        if old and self._identity_worker_mode!=self._identity_mode:
            old.close()
            self._identity_worker=None
        self._identity_step=None
        self._identity_cache={}
        self._supplied_observations={}

    def supply_relook_geometry(self, fid, step, source_rgb, live_rgb, source_geometry, live_geometry):
        """Supply same-world measured (depth,K,c2w) for a single pair before scoring."""
        def stamp(rgb):
            array=np.asarray(rgb)
            if array.ndim!=3 or array.shape[2]!=3 or array.dtype!=np.uint8:
                raise ValueError("geometry requires the exact uint8 source RGB")
            return (array.shape,hashlib.sha256(array.tobytes()).digest())
        for key in list(self._supplied_observations):
            if key[1]!=step:
                self._supplied_observations.pop(key)
        self._supplied_observations[(fid,step)]=(stamp(source_rgb),stamp(live_rgb),source_geometry,live_geometry)
        self._identity_cache.clear()

    def _identity_observations(self,fid,ref,obs,boxes):
        supplied=self._supplied_observations.get((fid,obs.step_count))
        if supplied is not None:
            source=self._source(fid,ref.roi)
            if source is None:
                return None
            rgb,box=source
            srck,livek,srcgeo,livegeo=supplied
            def matches(image,stamp):
                return image.shape==stamp[0] and hashlib.sha256(image.tobytes()).digest()==stamp[1]
            if not matches(rgb,srck) or not matches(obs.rgb,livek):
                return None
            width=obs.rgb.shape[1]
            return (rgb,[box],*srcgeo),(obs.rgb,np.asarray(boxes)*width/WORK_WIDTH,*livegeo)
        # The last keyframe is not a substitute for current-camera geometry.
        info=self._last_feed_info
        current=info.get("frame_id")
        if not info.get("is_keyframe") or current is None:
            return None
        scale=self._metric_scale_value()
        snap=self._map_snapshot()
        source=self._source(fid,ref.roi)
        if not source:
            return None
        src_rgb,src_box=source
        native=src_rgb if current==fid else _rgb(self.client.get_frame_image(current)[1])
        source=mapped_frame(snap.get(fid),src_rgb,src_rgb,[src_box],scale)
        w=obs.rgb.shape[1]
        live=mapped_frame(snap.get(current),native,obs.rgb,np.asarray(boxes)*w/WORK_WIDTH,scale)
        return (source,live) if source is not None and live is not None else None

    def _stage3_decide(self,source,live):
        if not getattr(self,"_identity_worker",None):
            self._identity_worker=Worker(Path(os.environ["NAV_ROOT"]),self._identity_mode,
                                         os.environ.get("NAV_STAGE3_DEVICE","cpu"))
            self._identity_worker_mode=self._identity_mode
        return self._identity_worker.decide(source,live)

    def _support(self,fid,ref,obs,boxes):
        if self._identity_mode=="original":
            return super()._support(fid,ref,obs,boxes)
        if self._identity_step!=obs.step_count:
            self._identity_cache.clear()
            self._identity_step=obs.step_count
        key=(fid,id(ref),tuple(map(tuple,boxes)))
        if key not in self._identity_cache:
            answer=(None,"identity_missing_geometry")
            if boxes:
                pair=self._identity_observations(fid,ref,obs,boxes)
                if pair is not None:
                    verdict=self._stage3_decide(*pair)
                    state,idx=verdict["state"],verdict["index"]
                    if state=="same" and type(idx) is int and 0<=idx<len(boxes):
                        answer=(boxes[idx],"learned_same_instance")
                    elif state in ("different","uncertain") and idx is None:
                        answer=(None,"identity_"+state)
                    else:
                        raise ValueError("invalid identity verdict")
            self._identity_cache[key]=answer
        return self._identity_cache[key]
