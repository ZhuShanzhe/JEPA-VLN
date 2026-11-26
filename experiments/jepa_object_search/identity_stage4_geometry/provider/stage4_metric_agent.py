"""Identical A/B adapter with private paired observations and a visual metric ruler."""
import json
import os
from pathlib import Path

import cv2
import numpy as np

from appearance import _rgb, WORK_WIDTH
from stage4_agent import Stage4GeometryAgent


class CausalStage4GeometryAgent(Stage4GeometryAgent):
    def _init_report_memory(self):
        super()._init_report_memory()
        self._geometry_ruler_context = False
        self._ruler_step = None
        self._ruler_record = {}

    def _visual_ruler(self, step):
        if self._ruler_step == step:
            return self._ruler_record
        self._ruler_step = step
        self._ruler_record = {"source":"unavailable","scale":None}
        native = getattr(self,"_metric_snapshot",{})
        if native.get("source") == "camera_height" and native.get("scale") and not native.get("pending_count"):
            self._ruler_record = {"source":"locked_camera_height","scale":float(native["scale"])}
            return self._ruler_record
        from agents import navigator
        snap = self._map_snapshot()
        ids = sorted(snap)[-12:]
        # align_R is gravity estimated by MOST from its predicted camera rotations.
        if len(ids)<6 or self.align_R is None:
            return self._ruler_record
        scales=[]
        for half in (ids[::2],ids[1::2]):
            grid = navigator.OccupancyGrid.from_frame_points([snap[k] for k in half],self.align_R,
                                                             cam_height_m=1.5,unit_per_m=None)
            if grid is None or not np.isfinite(grid.unit_per_m) or grid.unit_per_m<=0:
                return self._ruler_record
            scales.append(float(1/grid.unit_per_m))
        if max(scales)/min(scales) <= 1.15:
            self._ruler_record = {"source":"split_camera_height_1p5m","scale":float(np.mean(scales)),
                                  "independent_half_scales":scales,"map_frames":len(ids)}
        else:
            self._ruler_record.update(source="floor_rulers_disagree",independent_half_scales=scales)
        return self._ruler_record

    def _metric_scale_value(self):
        if getattr(self,"_geometry_ruler_context",False):
            return self._ruler_record.get("scale")
        return super()._metric_scale_value()

    def _geometry_log(self,obs,fid,kind,**extra):
        if kind=="geometry":
            extra["native_scale_source"]=getattr(self,"_metric_snapshot",{}).get("source")
            if not self._last_feed_info.get("is_keyframe") and self._geometry_mode=="anchored_pnp":
                extra["verification_ruler"]=self._ruler_record
        super()._geometry_log(obs,fid,kind,**extra)

    def _capture_geometry(self,fid,ref,obs,boxes):
        root=os.environ.get("NAV_STAGE4_PRIVATE_CAPTURE") or os.environ.get("NAV_STAGE4_PRIVATE_PAIRS")
        if not root:
            return
        target=Path(root)/f"step_{int(obs.step_count)}_source_{int(fid)}"
        if (target/"complete.json").is_file():
            return
        target.mkdir(parents=True,exist_ok=True,mode=0o700)
        snap=self._map_snapshot()
        ids=list(dict.fromkeys([fid,*sorted(snap,reverse=True)[:2]]))
        def save_rgb(name,rgb):
            if not cv2.imwrite(str(target/name),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):
                raise OSError("cannot store geometry capture")
        save_rgb("live.png",np.asarray(obs.rgb))
        captured=[]
        for key in ids:
            if key not in snap:
                continue
            meta,jpeg=self.client.get_frame_image(key)
            if not meta.get("found") or not jpeg:
                continue
            native=_rgb(jpeg)
            if native is None:
                continue
            frame=snap[key]
            save_rgb(f"anchor_{key}.png",native)
            np.savez_compressed(target/f"anchor_{key}.npz",**{k:frame[k] for k in ("points","colors","rows","pose","frame_id")})
            captured.append(int(key))
        row=dict(source=int(fid),anchors=captured,source_normalized_roi=list(map(float,ref.roi)),
                 live_boxes_work_width=np.asarray(boxes).tolist(),work_width=WORK_WIDTH,step=int(obs.step_count),
                 is_keyframe=bool(self._last_feed_info.get("is_keyframe")),ruler=self._ruler_record,
                 native_scale_source=getattr(self,"_metric_snapshot",{}).get("source"),
                 native_scale=super()._metric_scale_value())
        temp=target/"partial.json"
        temp.write_text(json.dumps(row,allow_nan=False))
        temp.replace(target/"complete.json")

    def _identity_observations(self,fid,ref,obs,boxes):
        is_live=not self._last_feed_info.get("is_keyframe")
        capturing=bool(os.environ.get("NAV_STAGE4_PRIVATE_CAPTURE") or os.environ.get("NAV_STAGE4_PRIVATE_PAIRS"))
        use_ruler=is_live and self._geometry_mode=="anchored_pnp"
        if use_ruler or capturing:
            self._visual_ruler(int(obs.step_count))
        if capturing:
            try:
                self._capture_geometry(fid,ref,obs,boxes)
            except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:
                self._geometry_log(obs,fid,"capture_unavailable",reason=type(exc).__name__)
        self._geometry_ruler_context=use_ruler
        try:
            return super()._identity_observations(fid,ref,obs,boxes)
        finally:
            self._geometry_ruler_context=False
