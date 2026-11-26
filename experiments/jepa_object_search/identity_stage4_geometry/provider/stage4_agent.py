"""Provide live object-verification geometry while leaving navigation and SLAM policy intact."""
import json
import os
import time
from pathlib import Path

import numpy as np

from appearance import WORK_WIDTH, _rgb
from identity_agent import IdentityMemoryAgent, mapped_frame


class Stage4GeometryAgent(IdentityMemoryAgent):
    def _init_report_memory(self):
        super()._init_report_memory()
        self._geometry_mode = os.environ.get("NAV_STAGE4_GEOMETRY", "keyframes_only")
        if self._geometry_mode not in {"keyframes_only", "anchored_pnp"}:
            raise ValueError("unknown geometry policy")
        self._tracking_step = None
        self._tracked_observations = {}

    def _geometry_log(self, obs, fid, kind, **extra):
        path = os.environ.get("NAV_STAGE4_GEOMETRY_EVENTS")
        if path:
            row = dict(mode=self._geometry_mode, step=int(obs.step_count), source=int(fid),
                       event=kind, **extra)
            with Path(path).open("a", encoding="utf8") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")

    def _identity_observations(self, fid, ref, obs, boxes):
        ordinary = super()._identity_observations(fid, ref, obs, boxes)
        is_key = bool(self._last_feed_info.get("is_keyframe"))
        if ordinary is not None or is_key or self._geometry_mode == "keyframes_only":
            self._geometry_log(obs, fid, "geometry", keyframe=is_key, available=ordinary is not None,
                               reason="native_pair" if ordinary is not None else "no_native_pair")
            return ordinary
        from anchored_tracking import estimate_from_anchor
        from map_views import rgb_aligned, metric_view
        started = time.monotonic()
        scale = self._metric_scale_value()
        snap = self._map_snapshot() if scale is not None and np.isfinite(scale) and scale > 0 else {}
        src = self._source(fid, ref.roi) if fid in snap else None
        source = mapped_frame(snap[fid], src[0], src[0], [src[1]], scale) if src else None
        if source is None:
            self._geometry_log(obs, fid, "geometry", keyframe=False, available=False,
                               reason="invalid_source_or_metric_scale", have_map=bool(snap))
            return None
        if self._tracking_step != obs.step_count:
            self._tracking_step = obs.step_count
            self._tracked_observations.clear()
        # Map comes from the running agent's current observation callback. Frame ids in
        # this snapshot have already been committed; no frame after the callback is read.
        anchors = list(dict.fromkeys([*sorted(snap, reverse=True)[:2], fid]))
        live_rgb = np.asarray(obs.rgb)
        live_boxes = np.asarray(boxes, dtype=float) * live_rgb.shape[1] / WORK_WIDTH
        attempts = []
        for anchor in anchors:
            if anchor not in self._tracked_observations:
                if anchor == fid:
                    native = src[0]
                else:
                    meta, jpeg = self.client.get_frame_image(anchor)
                    native = _rgb(jpeg) if jpeg and meta.get("found") else None
                geometry, record = None, {"reason": "anchor_image_or_grid_missing"}
                if native is not None and rgb_aligned(snap[anchor], native, 8):
                    view = metric_view(snap[anchor], scale, 8)
                    if view is not None:
                        geometry, record = estimate_from_anchor(native, view, live_rgb)
                self._tracked_observations[anchor] = geometry, record
            geometry, record = self._tracked_observations[anchor]
            entry = dict(anchor=int(anchor), **record)
            if geometry is None:
                attempts.append(entry)
                continue
            depth, K, pose = geometry
            counts = []
            for x0, y0, x1, y1 in live_boxes:
                patch = depth[max(0,int(y0)):int(np.ceil(y1)),max(0,int(x0)):int(np.ceil(x1))]
                counts.append(int(np.isfinite(patch).sum()))
            entry["box_depth_points"] = counts
            attempts.append(entry)
            # The frozen verifier requires all competing boxes to be scorable.
            if counts and min(counts) >= 9:
                self._geometry_log(obs, fid, "geometry", keyframe=False, available=True,
                                   reason="anchored_track", metric_scale=float(scale), attempts=attempts,
                                   milliseconds=round(1000*(time.monotonic()-started),1))
                return source, (live_rgb, live_boxes, depth, K, pose)
        self._geometry_log(obs, fid, "geometry", keyframe=False, available=False, reason="no_supported_anchor",
                           metric_scale=float(scale), attempts=attempts, milliseconds=round(1000*(time.monotonic()-started),1))
        return None

    def _support(self, fid, ref, obs, boxes):
        result = super()._support(fid, ref, obs, boxes)
        self._geometry_log(obs, fid, "support", boxes=len(boxes), accepted=result[0] is not None, reason=result[1])
        return result
