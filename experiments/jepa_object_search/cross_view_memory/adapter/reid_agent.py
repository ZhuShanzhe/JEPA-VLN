"""Optional grounded re-identification evidence for native duplicate review."""
import math
import os
from pathlib import Path

import numpy as np

from identity import relation
from model_worker import MatcherProcess
from regions import crop_evidence, heading_change_degrees
from report_agent import PROJECT, ReportFrontier


class ReIDFrontier(ReportFrontier):
    def _init_report_memory(self):
        super()._init_report_memory()
        self._region_cache = {}
        self._reid_error = False

    def _matcher(self):
        weights = os.environ.get("NAV_REID_WEIGHTS")
        if not weights or self._reid_error:
            return None
        if not hasattr(self, "_reid_matcher"):
            interpreter = os.environ.get("NAV_REID_PYTHON", str(
                PROJECT.parent / "jepa_instance_bench_20261016/policy_env/bin/python"))
            try:
                self._reid_matcher = MatcherProcess(
                    interpreter, PROJECT, Path(weights), os.environ.get("NAV_REID_DEVICE", "cpu"))
            except (OSError, ValueError, RuntimeError, EOFError) as exc:
                self._reid_error = True
                self._note("cross_view_worker_unavailable", None, reason=type(exc).__name__)
                return None
        return self._reid_matcher

    def _crop(self, record):
        if record.frame_id is None or record.bbox is None:
            return None
        if record.oid not in self._region_cache:
            meta, jpeg = self.client.get_frame_image(int(record.frame_id))
            if not meta.get("found") or not jpeg:
                return None
            value = crop_evidence(jpeg, record.bbox)
            # Bounded episode-local RGB evidence. SLAM poses are read live.
            if len(self._region_cache) >= 128:
                self._region_cache.pop(next(iter(self._region_cache)))
            self._region_cache[record.oid] = value
        return self._region_cache[record.oid]

    def _ingest_semantic_hits(self, observation, hits, select=True):
        if not os.environ.get("NAV_REID_WEIGHTS") or self._reid_error:
            return super()._ingest_semantic_hits(observation, hits, select)
        before = set(self.instance_store.observations)
        result = super()._ingest_semantic_hits(observation, hits, select)
        # Preserve the first grounded image when it becomes available even if
        # this mapper later discards the JPEG. A cached crop never confirms identity.
        for oid in sorted(self.instance_store.observations.keys() - before):
            try:
                self._crop(self.instance_store.get_observation(oid))
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                self._note("first_view_unavailable", None, observation_id=oid, reason=type(exc).__name__)
        return result

    def _reid_evidence(self, row):
        obs = self.instance_store.get_observation(row["observation_id"])
        if obs is None or obs.frame_id is None or self.instance_store.instance_for_observation(obs.oid):
            return None
        scale = self._metric_scale_value()
        if scale is None or not math.isfinite(scale) or scale <= 0 or self.align_R is None:
            return None
        # Check all registry nodes: compact native tool output drops the 4th neighbor.
        nearby = self.instance_store.nearby(obs.point, scale, self.instance_dup_radius_m,
                                            top_k=max(1, len(self.instance_store.nodes)))
        if len(nearby) != 1:
            return None
        distance, node = nearby[0]
        if not math.isfinite(distance) or distance > min(0.60, self.instance_dup_radius_m):
            return None
        # The native neighborhood uses floor-plane distance. Check height too
        # before hinting for stacked objects (e.g. two paintings on one wall).
        offset = np.asarray(node.point) - np.asarray(obs.point)
        if offset.shape != (3,) or not np.isfinite(offset).all() or np.linalg.norm(offset) * scale > 0.60:
            return None
        if node.iid not in {other["instance_id"] for other in row["neighbors"]}:
            return None
        records = self._records(node)
        if relation([obs], records, obs.evidence.get("goal_index"),
                    self._instance_goal_index.get(node.iid)) != "unknown":
            return None
        if not records:
            return None
        ref = min(records, key=lambda r: (r.step, r.oid))
        if ref.frame_id is None or ref.frame_id == obs.frame_id or ref.step > obs.step:
            return None
        pose1 = self.client.get_frame_pose(int(ref.frame_id))
        pose2 = self.client.get_frame_pose(int(obs.frame_id))
        yaw = heading_change_degrees(pose1, pose2, self.align_R)
        if yaw is None:
            return None
        source, candidate = self._crop(ref), self._crop(obs)
        if source is None or candidate is None:
            return None
        model = self._matcher()
        if model is None:
            return None
        pred = model.compare(source, candidate, yaw)
        score, threshold = float(pred["score"]), float(pred["threshold"])
        if not np.isfinite([score, threshold]).all() or not pred["match"] or score < threshold:
            return None
        return {
            "compare_with_instance": int(node.iid), "report_executed": bool(node.reported),
            "candidate_step": int(obs.step), "first_registration_step": int(ref.step),
            "view_change_deg": round(yaw, 1), "estimated_distance_m": round(distance, 2),
            "similarity": round(score, 4), "validation_threshold": round(threshold, 4),
            "guidance": ("Inspect both native views and geometry. If independent evidence confirms "
                         "a duplicate, resolve to the existing instance; preserve distinct objects as NEW."),
        }

    def _dup_review_payload(self):
        rows, images, omitted = super()._dup_review_payload()
        if not os.environ.get("NAV_REID_WEIGHTS") or self._reid_error:
            return rows, images, omitted
        for row in rows:
            try:
                hint = self._reid_evidence(row)
            except (OSError, ValueError, EOFError, RuntimeError, KeyError, TypeError) as exc:
                self._note("cross_view_evidence_unavailable", None, reason=type(exc).__name__)
                # Never let a corrupt observation or dead worker interrupt the native review.
                if isinstance(exc, (OSError, EOFError, RuntimeError)):
                    self._reid_error = True
                    model = getattr(self, "_reid_matcher", None)
                    if model is not None:
                        model.close()
                        del self._reid_matcher
                continue
            if hint:
                row["cross_view_evidence"] = hint
                self._note("cross_view_review_hint", hint["compare_with_instance"],
                           observation_id=row["observation_id"], similarity=hint["similarity"])
        return rows, images, omitted
