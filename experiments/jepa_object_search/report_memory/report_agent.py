"""Confirmed-action report memory for MOST multi-instance navigation."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(os.environ.get("NAV_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT / "jepa_confirm_v18/agent"))
from agents import navigator
from benchmark_api import Action
from confirm_agent import ConfirmFrontier
from appearance import build_reference, match_reference, match_registered_view, same_image_region
from exploration_memory import MemoryChannel
from identity import guarded_instance_memory, relation


@dataclass(frozen=True)
class PendingReport:
    step: int
    instance: int
    observations: tuple
    camera_anchor: object


class ReportFrontier(ConfirmFrontier):
    def _nav_reset_state(self):
        super()._nav_reset_state()
        self.instance_store = guarded_instance_memory()

    def _registered_spot_at(self, frame_id, pixel_norm):
        # This early native hook has only a click, not the proposal's mask or
        # identity. Leave replay detection to the store once it gets the box.
        return None

    def init_episode(self, observation):
        self._init_report_memory()
        super().init_episode(observation)

    def _init_report_memory(self):
        self._pending_report = None
        self._report_views = []
        self._last_auto_check = -1
        self._recheck_cursor = 0
        self._requested = set()
        self._last_view_check = None
        self._match_cache_step = None
        self._match_cache = {}

    def _build_decider_input(self, observation, local_map=False):
        state, image = super()._build_decider_input(observation, local_map)
        aliases = {}
        for node in self.instance_store.available():
            owner = self._known_owner(node)
            if owner is not None:
                aliases[node.iid] = owner.iid
        state = dict(state)
        if aliases:
            state["instances"] = [
                entry for entry in state.get("instances", []) if entry.get("id") not in aliases
            ]
        state["instance_report_memory"] = {
            "executed_ids": [n.iid for n in self.instance_store.nodes if n.reported],
            "same_evidence_as_executed": aliases,
            "awaiting_close_view": sorted(
                n.iid
                for n in self.instance_store.available()
                if n.iid in self._requested and n.iid not in aliases
            ),
            "instruction": "An executed report is an action record. Search for other instances, including nearby similar objects. Seek a closer or different view for unresolved clues.",
        }
        return state, image

    def _connect(self):
        super()._connect()
        self.channel = MemoryChannel(self.channel, self)

    def _records(self, node):
        return [
            r
            for oid in node.observation_ids
            if (r := self.instance_store.get_observation(oid)) is not None
        ]

    def _known_owner(self, node):
        if node.reported:
            return node
        records = self._records(node)
        goals = self._instance_goal_index
        for other in self.instance_store.nodes:
            if (
                other.reported
                and relation(records, self._records(other), goals.get(node.iid), goals.get(other.iid))
                == "same"
            ):
                return other
        return None

    def _collect_views(self):
        store = self.instance_store
        stamp = (self.now, len(store.observations), len(store.nodes))
        if self._last_view_check == stamp:
            return
        self._last_view_check = stamp
        for node in store.available():
            for record in self._records(node):
                oid, fid = record.oid, record.frame_id
                if fid is None or oid in self._views or self.now < self._retry_after.get(oid, -1):
                    continue
                if record.bbox is None:
                    continue
                meta, jpeg = self.client.get_frame_image(int(fid))
                if not meta.get("found") or not jpeg:
                    self._retry_after[oid] = self.now + 6
                    continue
                ref = build_reference(jpeg, record.bbox, point_1000=record.evidence.get("pixel_norm"))
                if ref is None:
                    self._retry_after[oid] = float("inf")  # Keyframe and box are immutable.
                    self._note("clue_without_usable_region", node.iid, frame=int(fid))
                    continue
                self._views[oid] = (int(fid), ref)
                self._endorsed.add(node.iid)
                self._note("object_region_saved", node.iid, frame=int(fid))

    def _report_gate(self, node, observation):
        self._collect_views()
        refs = self._references(node)
        if not refs:
            return False, "need_grounded_object_region"
        reason = "need_metric_near_view"
        for fid, ref in reversed(refs):
            geo = self._geometry(node, fid)
            # geo[2] is a map estimate to the object, not the judge's viewpoint distance.
            if geo is None or geo[2] > self.report_near_dist_m:
                continue
            # A distant source image cannot certify a permissive active-target
            # report; allow the broader native range at a known nearby viewpoint.
            near_capture = (
                geo[0] <= 0.35 and geo[1] <= self.report_near_dist_m and abs(geo[3]) <= math.radians(30)
            )
            if geo[2] > min(0.70, self.report_near_dist_m) and not near_capture:
                continue
            match, bound = self._live_match(fid, ref, observation)
            reason = match.reason
            if match.matched:
                if self._competing_identity(node, match.box, observation, bound):
                    reason = "other_instance_explains_same_live_region"
                    continue
                return True, "localized_object_and_near_pose"
        return False, reason

    def _live_match(self, fid, ref, observation):
        info = self._last_feed_info
        bound = bool(info.get("is_keyframe")) and info.get("frame_id") == fid
        if self._match_cache_step != observation.step_count:
            self._match_cache_step = observation.step_count
            self._match_cache.clear()
        key = (id(ref), bound)
        if key not in self._match_cache:
            self._match_cache[key] = (
                match_registered_view(ref, observation.rgb)
                if bound
                else match_reference(ref, observation.rgb)
            )
        return self._match_cache[key], bound

    def _competing_identity(self, node, box, observation, bound):
        for other in self.instance_store.nodes:
            if other.iid == node.iid:
                continue
            dist = self._instance_distance_m(other)
            if dist is not None and dist > self.report_near_dist_m + 1.5:
                continue
            for fid, ref in self._references(other):
                if bound and fid != self._last_feed_info.get("frame_id"):
                    continue
                rival, _ = self._live_match(fid, ref, observation)
                if rival.matched and same_image_region(box, rival.box):
                    if (
                        relation(
                            self._records(node),
                            self._records(other),
                            self._instance_goal_index.get(node.iid),
                            self._instance_goal_index.get(other.iid),
                        )
                        == "same"
                    ):
                        continue
                    return True
        return False

    def _capture_report_view(self):
        here = self._estimated_current_pose()
        scale = self._metric_scale_value()
        if (
            here is None
            or scale is None
            or not math.isfinite(scale)
            or scale <= 0
            or self.align_R is None
        ):
            return None
        info = self._last_feed_info
        for fid in dict.fromkeys(
            info.get(k) for k in ("frame_id", "viewable_frame_id", "last_available_frame_id")
        ):
            if fid is None:
                continue
            frame = self.client.get_frame_pose(int(fid))
            if frame is None:
                continue
            x, y, yaw = navigator.pose_to_yaw_2d(frame, self.align_R)
            dx, dy = (here[0] - x) * scale, (here[1] - y) * scale
            if np.isfinite([dx, dy, yaw, here[2]]).all() and math.hypot(dx, dy) <= 1.0:
                return (
                    int(fid),
                    dx * math.cos(yaw) + dy * math.sin(yaw),
                    -dx * math.sin(yaw) + dy * math.cos(yaw),
                    math.remainder(here[2] - yaw, 2 * math.pi),
                )
        return None

    def _reported_views_for_planning(self):
        scale = self._frontier_scale
        if scale is None or not math.isfinite(scale) or scale <= 0 or self.align_R is None:
            return []
        result = []
        for fid, forward, left, angle in self._report_views:
            frame = self.client.get_frame_pose(fid)
            if frame is not None:
                x, y, yaw = navigator.pose_to_yaw_2d(frame, self.align_R)
                result.append(
                    (
                        x + (math.cos(yaw) * forward - math.sin(yaw) * left) / scale,
                        y + (math.sin(yaw) * forward + math.cos(yaw) * left) / scale,
                        yaw + angle,
                    )
                )
        return result

    def _unreported_points_for_planning(self):
        points = []
        for node in self.instance_store.available():
            if self._known_owner(node) is not None:
                continue
            xy = tuple(map(float, node.point[:2]))
            if len(xy) != 2 or not np.isfinite(xy).all():
                return None
            points.append(xy)
        # A not-yet-localized duplicate proposal could be a new physical instance.
        proposals = getattr(self, "_proposal_queue", {}).values()
        if getattr(self, "_dup_ledger", {}) or any(
            p.get("status") in ("pending", "uncertain", "duplicate_review") for p in proposals
        ):
            return None
        return points

    def _stage_report(self, node, observation, reason):
        self._pending_report = PendingReport(
            int(observation.step_count),
            node.iid,
            tuple(node.observation_ids),
            self._capture_report_view(),
        )
        self._note("report_dispatched", node.iid, reason=reason)
        return int(Action.TARGET_FOUND)

    def _settle_report(self, observation):
        pending = self._pending_report
        step = int(observation.step_count)
        if pending is None or step <= pending.step:
            return
        self._pending_report = None
        if step != pending.step + 1 or observation.previous_action != int(Action.TARGET_FOUND):
            self._note("report_not_acknowledged", pending.instance)
            self._stamp_pending_outcome(
                "unconfirmed:report_not_observed",
                "No consecutive observed report action. This instance remains available.",
            )
            return
        owners = {
            owner.iid: owner
            for oid in pending.observations
            if (owner := self.instance_store.instance_for_observation(oid)) is not None
        }
        if len(owners) != 1:
            self._note("report_owner_unresolved", pending.instance)
            return
        node = next(iter(owners.values()))
        if self._known_owner(node) is not None:
            self._note("report_already_recorded", node.iid)
            return
        claim = self.instance_store.claim(node, step=pending.step, observation_ids=pending.observations)
        if claim is None:
            return
        self._reported_count += 1
        self._no_hit_queries = 0
        goal = self._instance_goal_index.get(node.iid)
        if goal is not None:
            self._goal_found.add(int(goal))
        if pending.camera_anchor is not None:
            self._report_views.append(pending.camera_anchor)
        self._requested.difference_update((node.iid, pending.instance))
        if self.target_instance_id in (node.iid, pending.instance):
            self.mode = "reported"
            self._scanning = False
        if self._confirm_pending in (node.iid, pending.instance):
            self._release("acknowledged_report")
        self._note("report_action_acknowledged", node.iid, source_step=pending.step)

    def _feed_frame(self, observation):
        self._settle_report(observation)
        return super()._feed_frame(observation)

    def _continue_other_target(self, observation):
        if self.mode == "nav" and self.target_instance_id is not None and self.follower is not None:
            action, arrived, stuck = super()._nav_action(observation)
            if stuck:
                return self._nav_failed_recovery(observation)
            if arrived:
                return self._handle_arrival_transition(observation)
            if action is not None:
                return action
        return int(Action.TURN_LEFT)

    def _report_found(self, instance_id=None):
        node = self.instance_store.get(instance_id)
        observation = self._last_observation
        if node is not None and self._pending_report is not None:
            if (
                self._pending_report.step == observation.step_count
                and self._pending_report.instance == node.iid
            ):
                return int(Action.TARGET_FOUND)
            return int(Action.TURN_LEFT)
        if node is None or self._known_owner(node) is not None:
            self._stamp_pending_outcome(
                "blocked:report_unavailable", "Unknown instance or the same instance already reported."
            )
            if node is not None and self.target_instance_id == node.iid:
                self._clear_current_target()
                self.mode = "explore"
            return self._continue_other_target(observation)
        self._requested.add(node.iid)
        ok, reason = self._report_gate(node, observation)
        if ok:
            return self._stage_report(node, observation, reason)
        self._note("report_deferred", node.iid, reason=reason)
        self._stamp_pending_outcome(
            "deferred:need_close_object_view", "Keep the clue. Approach and verify it in a live view."
        )
        if (
            self._confirm_pending is None
            and not self._arrival_transition_active
            and self.target_instance_id in (None, node.iid)
        ):
            action = self._return(observation, [node])
            if action is not None:
                return action
        if self.target_instance_id not in (None, node.iid):
            return self._continue_other_target(observation)
        return int(Action.TURN_LEFT)

    def _auto_report(self, observation):
        if self._pending_report or self._last_auto_check == observation.step_count:
            return None
        self._last_auto_check = observation.step_count
        self._collect_views()
        # Only canonical task instances, already grounded in an object region.
        candidates = [
            n
            for n in self.instance_store.available()
            if self._references(n) and self._known_owner(n) is None
        ]
        if not candidates:
            return None
        index = self._recheck_cursor % len(candidates)
        self._recheck_cursor += 1
        # An active / explicitly requested object goes first, while the rotating
        # slot lets another visible object trigger a side report during navigation.
        preferred = next((n for n in candidates if n.iid == self.target_instance_id), None)
        if preferred is None:
            preferred = next((n for n in candidates if n.iid in self._requested), None)
        for node in (
            [preferred] if preferred is not None and preferred is not candidates[index] else []
        ) + [candidates[index]]:
            if getattr(self, "_goal_images", None) and self._instance_goal_index.get(node.iid) is None:
                continue
            ok, reason = self._report_gate(node, observation)
            if ok:
                return self._stage_report(node, observation, reason)
        return None

    def _nav_action(self, observation):
        action = self._auto_report(observation)
        if action is not None:
            return action, False, False
        return super()._nav_action(observation)

    def _explore_action(self, observation):
        action = self._auto_report(observation)
        return action if action is not None else super()._explore_action(observation)

    def _return(self, observation, nodes):
        return super()._return(
            observation, [n for n in nodes if n is not None and self._known_owner(n) is None]
        )

    def _execute_decider_result(self, observation, result, event, images=None):
        if result.action == "GOTO_INSTANCE":
            node = self.instance_store.get(result.target_id)
            if node is not None and self._known_owner(node) is not None:
                self._stamp_pending_outcome(
                    "blocked:already_reported", "Select another instance or a fresh frontier."
                )
                self._note("repeat_visit_avoided", node.iid)
                if self.target_instance_id == node.iid:
                    self._clear_current_target()
                    self.mode = "explore"
                if self.mode == "nav" and self.target_instance_id is not None:
                    return self._continue_other_target(observation)
                return self._autonomous_explore_action(observation)
        return super()._execute_decider_result(observation, result, event, images)

    def _tool_resolve_duplicate(self, observation_id, decision, duplicate_of=None, text=""):
        obs = self.instance_store.get_observation(observation_id)
        other = self.instance_store.get(duplicate_of)
        if str(decision).strip().upper() == "DUPLICATE" and obs is not None and other is not None:
            if (
                relation(
                    [obs],
                    self._records(other),
                    obs.evidence.get("goal_index"),
                    self._instance_goal_index.get(other.iid),
                )
                == "distinct"
            ):
                self._note("protected_separate_observation", other.iid)
                return {
                    "error": "Separate target bindings or nonoverlapping objects in the same view. Resolve as NEW or choose another identity."
                }
        return super()._tool_resolve_duplicate(observation_id, decision, duplicate_of, text)

    def _tool_merge_instances(self, instance_id, other_instance_id, text=""):
        left, right = self.instance_store.get(instance_id), self.instance_store.get(other_instance_id)
        goals = self._instance_goal_index
        if (
            left is not None
            and right is not None
            and relation(
                self._records(left), self._records(right), goals.get(left.iid), goals.get(right.iid)
            )
            == "distinct"
        ):
            return {"error": "Independent instances have conflicting identity evidence. Keep both."}
        result = super()._tool_merge_instances(instance_id, other_instance_id, text)
        if "merged" in result:
            self._reported_count = self.instance_store.count_reported()
        return result
