"""Re-present an unresolved cross-view pair when the map acquires metric scale."""
import math
import os
from hashlib import sha256

import numpy as np

from identity import relation


class DelayedPoseMixin:
    scale_wait_steps = 4

    def _init_report_memory(self):
        super()._init_report_memory()
        self._scale_waiting = {}
        self._scale_delivered = {}

    def _review_signature(self, obs, neighbors, scale):
        """Existing measured images are indexed by frame ID; the map can change at any step."""
        key = (scale, tuple((n.iid, bool(n.reported), float(d)) for d,n in neighbors))
        if len(neighbors) != 1:
            return key
        node = neighbors[0][1]
        records = self._records(node)
        fids = {r.frame_id for r in [obs, *records] if r.frame_id is not None}
        local_records = tuple(sorted((r.oid, r.frame_id, tuple(r.bbox) if r.bbox is not None else (),
                                     (r.evidence or {}).get("goal_index"))
                                    for r in self.instance_store.observations.values() if r.frame_id in fids))
        snapshots = self._map_snapshot()
        digest = sha256()
        for fid in sorted(fids):
            snap = snapshots.get(fid) or {}
            for field in ("pose", "rows", "points", "colors"):
                if field in snap:
                    data = np.ascontiguousarray(snap[field])
                    digest.update(str((fid, field, data.shape, data.dtype.str)).encode())
                    digest.update(memoryview(data).cast("B"))
        return key + (tuple(r.oid for r in records), self._instance_goal_index.get(node.iid),
                      local_records, digest.digest())

    def _usable_scale(self):
        scale = self._metric_scale_value()
        return scale if scale is not None and math.isfinite(scale) and scale > 0 else None

    def _hold_cross_view(self, row):
        oid = row["observation_id"]
        ledger = self._dup_ledger.get(oid)
        weights = os.environ.get("NAV_REID_POSE_WEIGHTS")
        if (not weights or self._pose_failed or ledger is None or self._usable_scale() is not None
                or self.now - ledger["first_step"] >= self.scale_wait_steps):
            return False
        obs = self.instance_store.get_observation(oid)
        neighbors = ledger["neighbors"]
        if (len(neighbors) != 1 or obs is None or obs.frame_id is None or obs.bbox is None
                or self.instance_store.instance_for_observation(oid) is not None):
            return False
        node = self.instance_store.get(neighbors[0]["instance_id"])
        records = self._records(node) if node is not None else []
        if not records or relation([obs], records, (obs.evidence or {}).get("goal_index"),
                                   self._instance_goal_index.get(node.iid)) != "unknown":
            return False
        ref = min(records, key=lambda r: (r.step, r.oid))
        if ref.frame_id is None or ref.frame_id == obs.frame_id or ref.bbox is None or ref.step > obs.step:
            return False
        if oid not in self._scale_waiting:
            self._scale_waiting[oid] = ledger["first_step"]
            self._audit("pose_deferred_for_scale", observation=oid)
        return True

    def _tool_commit_candidates(self, reviews, label=""):
        out = super()._tool_commit_candidates(reviews, label)
        ready, waiting = [], []
        for row in out.get("duplicate_review", []):
            if self._hold_cross_view(row):
                waiting.append(row["observation_id"])
            else:
                ready.append(row)
        if waiting:
            out["duplicate_review"] = ready
            out["deferred_until_map_scale"] = waiting
            out["next"] = (out.get("next", "") + " Deferred observations are saved. Continue navigation now; "
                           "only rows in duplicate_review require immediate review. "
                           "Unscaled neighbor distances are not metres.")
        return out

    def _release_row(self, row):
        """Prepare one advisory. The native ledger owns all registration mutations."""
        oid = row["observation_id"]
        scale = self._usable_scale()
        out = dict(row)
        if scale is None:
            out["neighbors"] = [{k:v for k,v in n.items() if k != "dist_m"} for n in row["neighbors"]]
            out["metric_review"] = "Scale unavailable after waiting; inspect visual evidence and resolve normally."
            self._audit("pose_wait_expired", observation=oid)
        else:
            obs = self.instance_store.get_observation(oid)
            neighbors = self.instance_store.nearby(obs.point, scale, self.instance_dup_radius_m, top_k=4)
            out["neighbors"] = [dict(instance_id=n.iid, dist_m=round(float(d), 3), reported=bool(n.reported))
                                for d,n in neighbors]
            # Keep the native pending ledger consistent with the rows that are
            # displayed, including a neighbor disappearing after calibration.
            self._dup_ledger[oid]["neighbors"] = list(out["neighbors"])
            hint = None
            if os.environ.get("NAV_REID_POSE_WEIGHTS") and not self._pose_failed:
                try:
                    key = self._review_signature(obs, neighbors, scale)
                    cached = self._scale_delivered.get(oid)
                    if cached is not None and cached[0] == key:
                        hint = cached[1]
                    else:
                        hint = self._pose_hint(out, os.environ["NAV_REID_POSE_WEIGHTS"])
                        self._scale_delivered[oid] = key, hint
                except (OSError, RuntimeError, EOFError, ValueError, KeyError, ArithmeticError) as exc:
                    self._audit("deferred_pose_unavailable", observation=oid, reason=type(exc).__name__)
            if hint is not None:
                out["pose_jepa_review"] = hint
            out["metric_review"] = "Metric neighborhood refreshed. Inspect identity before linking."
            self._audit("pose_deferred_review_released", observation=oid, hint=bool(hint))
        return out

    def _build_decider_input(self, observation, local_map=False):
        state, images = super()._build_decider_input(observation, local_map)
        pending = {r["observation_id"]: r for r in state.get("duplicate_review_pending", [])}
        waiting = []
        for oid in list(self._scale_waiting):
            ledger = self._dup_ledger.get(oid)
            if ledger is None or self.instance_store.instance_for_observation(oid) is not None:
                self._scale_waiting.pop(oid)
                self._scale_delivered.pop(oid, None)
                pending.pop(oid, None)
                continue
            if self._hold_cross_view(ledger):
                pending.pop(oid, None)
                waiting.append(oid)
                continue
            pending[oid] = self._release_row(pending.get(oid, ledger))
        if waiting:
            state["deferred_until_map_scale"] = waiting
        state["duplicate_review_pending"] = list(pending.values())
        return state, images

    def _tool_resolve_duplicate(self, observation_id, decision, duplicate_of=None, text=""):
        obs = self.instance_store.get_observation(observation_id)
        oid = obs.oid if obs is not None else None
        row = self._dup_ledger.get(oid)
        if row is not None and str(decision).strip().upper() == "DUPLICATE":
            if self._hold_cross_view(row):
                return dict(status="waiting_for_map_scale", observation_id=oid,
                            next="Observation saved. Continue navigation for a fresh metric view; no retry this step.")
            if oid in self._scale_waiting and self._usable_scale() is not None:
                before = self._scale_delivered.get(oid)
                refreshed = self._release_row(row)
                if not any(str(n["instance_id"]) == str(duplicate_of) for n in refreshed["neighbors"]):
                    return dict(error="The requested identity is outside the current calibrated neighborhood.",
                                duplicate_review=[refreshed], next="Keep this observation pending or resolve as NEW.")
                if before != self._scale_delivered.get(oid):
                    return dict(status="calibrated_review_available", duplicate_review=[refreshed],
                                next="Read the refreshed evidence before resolving this observation.")
        out = super()._tool_resolve_duplicate(observation_id, decision, duplicate_of, text)
        if out.get("resolved"):
            self._scale_waiting.pop(oid, None)
            self._scale_delivered.pop(oid, None)
        return out
