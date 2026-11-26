"""Finish local confirmation on one observation before returning the next action."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import cv2
import numpy as np

from appearance import WORK_WIDTH, _rgb
from stage4_metric_agent import CausalStage4GeometryAgent
from optics import skew_view
from map_views import rgb_aligned
from shared_anchor_pair import windows_with_anchors


class BarrierAgent(CausalStage4GeometryAgent):
    def _init_report_memory(self):
        super()._init_report_memory()
        self._close_barrier()
        self._barrier_temp = None
        self._past_views = {}
        self._barrier_step = None
        self._barrier_cache = {}
        self._barrier_bundles = {}
        self._polled_step = None

    def _close_barrier(self):
        temp = getattr(self, "_barrier_temp", None)
        if temp is not None:
            temp.cleanup()
            self._barrier_temp = None

    def act(self, observation):
        # act returns one proposed primitive. The environment cannot execute it
        # while the report gate and the local worker below are still computing.
        try:
            action = super().act(observation)
        finally:
            self._close_barrier()
            self._barrier_bundles.clear()
        step = int(observation.step_count)
        info = self._last_feed_info or {}
        frame_id = info.get("frame_id") if info.get("is_keyframe") else None
        rgb = _rgb(observation.rgb)
        if rgb is not None:
            self._past_views[step] = (frame_id, rgb.copy())
        for old in list(self._past_views):
            if old < step - 63 or old > step:
                del self._past_views[old]
        return action

    def _barrier_work(self, stage, folder, deadline):
        root = Path(os.environ["NAV_ROOT"])
        env = "mapping_runtime" if stage == "reconstruct" else "b_verification"
        command = [str(root / env / "env/bin/python"), "-B", str(Path(__file__).with_name("barrier_worker.py")), stage, str(folder)]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("stationary identity computation deadline exceeded")
        log_root = os.environ.get("NAV_STAGE4_GEOMETRY_EVENTS")
        # Persist engineering diagnostics on the private run, never in released aggregates.
        error_log = Path(log_root).with_name("stage5_worker.log") if log_root else folder / "worker.log"
        with error_log.open("ab") as output:
            subprocess.run(command, check=True, timeout=remaining, stdout=output, stderr=subprocess.STDOUT,
                           env={**os.environ, "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1",
                                "PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True",
                                "MKL_NUM_THREADS":"2", "OMP_NUM_THREADS":"2"})

    def _stationary_verdict(self, fid, ref, obs, boxes):
        step = int(obs.step_count)
        snap = self._map_snapshot()
        scale = self._visual_ruler(step).get("scale")
        src = self._source(fid, ref.roi) if fid in snap else None
        def withheld(reason):
            return {"result":"withheld", "reason":reason}
        if src is None or not scale or not np.isfinite(scale) or scale <= 0:
            return withheld("source_map_or_scale_unavailable")
        if skew_view(snap[fid], scale) is None:
            return withheld("source_projection_unreliable")
        by_frame = {frame: k for k, (frame, _) in self._past_views.items() if frame is not None and k < step}
        native = {}
        for key in sorted(snap, reverse=True):
            if key == fid or key not in by_frame or skew_view(snap[key], scale) is None:
                continue
            meta, data = self.client.get_frame_image(key)
            rgb = _rgb(data) if meta.get("found") and data else None
            if rgb is not None and rgb.shape == src[0].shape:
                native[key] = rgb
            if len(native) == 2:
                break
        if step < 8 or len(native) < 2:
            return withheld("missing_two_visible_map_anchors")
        steps = windows_with_anchors(step, [by_frame[k] for k in native])
        if any(k not in self._past_views for k in steps):
            return withheld("causal_rgb_window_incomplete")
        signature = (fid, tuple(steps), tuple(native), float(scale))
        if self._barrier_temp is None:
            project = Path(os.environ["NAV_ROOT"])
            spool = project / "identity_stage5_confirmation/private"
            spool.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._barrier_temp = tempfile.TemporaryDirectory(prefix="confirm_", dir=spool)
        size = src[0].shape[1::-1]
        if signature not in self._barrier_bundles:
            folder = Path(self._barrier_temp.name) / f"bundle_{len(self._barrier_bundles)}"
            folder.mkdir(mode=0o700)
            resize = lambda rgb: cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            images = [src[0], *[resize(self._past_views[k][1]) for k in steps], resize(_rgb(obs.rgb))]
            np.save(folder / "images.npy", np.stack(images))
            for key in [fid, *native]:
                fields = {k:snap[key][k] for k in ("points", "rows", "pose", "frame_id")}
                np.savez(folder / f"map_{key}.npz", **fields)
            for key, rgb in native.items():
                np.save(folder / f"rgb_{key}.npy", rgb)
            self._barrier_work("reconstruct", folder, self._barrier_deadline)
            self._barrier_bundles[signature] = folder
        folder = self._barrier_bundles[signature]
        if (folder / "unobservable.json").exists():
            return withheld("unobservable_motion_in_past")
        query = folder / f"query_{len(self._barrier_cache)}"
        query.mkdir(mode=0o700)
        metadata = {"source":int(fid), "scale":float(scale), "source_box":list(map(float, src[1])),
                    "boxes":(np.asarray(boxes, float) * size[0] / WORK_WIDTH).tolist(),
                    "anchors":[(int(key), steps.index(by_frame[key]) + 1) for key in native]}
        (query / "query.json").write_text(json.dumps(metadata, allow_nan=False))
        self._barrier_work("score", query, self._barrier_deadline)
        return json.loads((query / "verdict.json").read_text())

    def _ready_native(self, fid, ref, obs, boxes):
        """One bounded poll on a *selected* current image; no future feeds."""
        info = self._last_feed_info or {}
        current = info.get("frame_id")
        if not info.get("is_keyframe") or current is None:
            return None
        step = int(obs.step_count)
        if self._polled_step != step:
            self._polled_step = step
            end = min(self._barrier_deadline, time.monotonic() + .75)
            while current not in self._map_snapshot() and time.monotonic() < end:
                time.sleep(min(.15, max(0.,end-time.monotonic())))
                self._map_step = None
        snap = self._map_snapshot()
        if current not in snap or fid not in snap:
            return None
        scale = self._visual_ruler(step).get("scale")
        source = self._source(fid, ref.roi)
        if source is None or not scale or not np.isfinite(scale) or scale <= 0:
            return None
        rgb, box = source
        meta, data = self.client.get_frame_image(current)
        native = _rgb(data) if meta.get("found") and data else None
        if native is None or not rgb_aligned(snap[current], native, 8) or not rgb_aligned(snap[fid], rgb, 8):
            return None
        query = cv2.resize(_rgb(obs.rgb), native.shape[1::-1], interpolation=cv2.INTER_AREA)
        if np.abs(query.astype(np.float32)-native).mean() >= 10.:
            return None
        def observation(key, image, regions):
            view = skew_view(snap[key], scale)
            if view is None:
                return None
            z = (view["grid"]-view["c2w"][:3,3]) @ view["c2w"][:3,2]
            depth = np.full(image.shape[:2], np.nan, dtype=np.float32)
            if z.shape != depth[::8,::8].shape:
                return None
            depth[::8,::8] = np.where((z>.05)&(z<30.), z, np.nan)
            for x,y,u,v in regions:
                patch = depth[max(0,int(y)):int(np.ceil(v)),max(0,int(x)):int(np.ceil(u))]
                if np.isfinite(patch).sum() < 9:
                    return None
            return image, regions, depth, view["K"], view["c2w"]
        src = observation(fid,rgb,[box])
        live = observation(current,query,np.asarray(boxes)*native.shape[1]/WORK_WIDTH)
        if src is None or live is None:
            return None
        verdict = self._stage3_decide(src,live)
        return {"result":verdict["state"],"reason":"current_native_pair_ready","selected_box":verdict["index"]}

    def _support(self, fid, ref, obs, boxes):
        native = super()._support(fid, ref, obs, boxes)
        if self._identity_mode != "jepa" or not boxes or native[1] != "identity_missing_geometry":
            return native
        step = int(obs.step_count)
        if self._barrier_step != step:
            self._close_barrier()
            self._barrier_step = step
            self._barrier_cache.clear()
            self._barrier_bundles.clear()
            self._barrier_deadline = time.monotonic() + 180.
        key = (int(fid), tuple(map(float, ref.roi)), tuple(map(tuple, boxes)))
        if key not in self._barrier_cache:
            start = time.monotonic()
            self._geometry_log(obs, fid, "stationary_begin")
            try:
                result = self._ready_native(fid, ref, obs, boxes)
                if result is None:
                    result = self._stationary_verdict(fid, ref, obs, boxes)
            except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
                self._geometry_log(obs, fid, "stationary_error", error=type(exc).__name__)
                # Do not return a motion or treat operational failures as misses.
                raise RuntimeError("stationary identity worker failed before action release") from exc
            state = result["result"]
            index = result.get("selected_box")
            self._geometry_log(obs, fid, "stationary_end", state=state, selected_box=index,
                               reason=result["reason"], seconds=round(time.monotonic()-start, 3))
            if state == "same" and type(index) is int and 0 <= index < len(boxes):
                verdict = boxes[index], "learned_same_instance"
            elif state in {"different", "uncertain", "withheld"} and index is None:
                verdict = None, "identity_" + (result["reason"] if state == "withheld" else state)
            else:
                raise ValueError("invalid stationary identity result")
            self._barrier_cache[key] = verdict
        return self._barrier_cache[key]
