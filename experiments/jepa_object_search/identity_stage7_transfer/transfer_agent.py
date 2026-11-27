"""Use the native route to a named distant target and its explicit report decision."""
import math
from contextlib import contextmanager

import numpy as np
from approach_agent import ApproachAgent
from appearance import WORK_WIDTH, _rgb
from identity import relation
from local_match import prepare_reference, match_prepared
from report_evidence import eligible_boxes
from visual_witness import MISSING_GEOMETRY


class ArrivalTransferAgent(ApproachAgent):
    def _init_report_memory(self):
        super()._init_report_memory()
        self._arrival_request = None
        self._native_leg = None
        self._routing_native_goto = False
        self._arrival_templates = {}

    @contextmanager
    def _report_intent(self, obs, result, event):
        prior = self._arrival_request
        node = self.instance_store.get(result.target_id)
        target = node.iid if node is not None else None
        active = self.instance_store.get(self.target_instance_id)
        self._arrival_request = (int(obs.step_count), target, event,
                                 active is not None and active.iid == target)
        try:
            yield
        finally:
            self._arrival_request = prior

    def _execute_decider_result(self, observation, result, event, images=None):
        if result.action == "REPORT_FOUND":
            with self._report_intent(observation,result,event):
                return super()._execute_decider_result(observation,result,event,images)
        self._native_leg = None
        if result.action == "GOTO_INSTANCE" and self._pending_report is None:
            node = self.instance_store.get(result.target_id)
            if (node is not None and self._known_owner(node) is None and
                    (not getattr(self,"_goal_images",None) or self._instance_goal_index.get(node.iid) is not None)):
                d = self._instance_distance_m(node)
                if d is not None and math.isfinite(d) and d > 1.8:
                    self._release_pursuit(observation.step_count,"native_target_plan",pause=False)
                    if self._confirm_view_state:
                        self._end_confirm_view("native_target_plan",observation.step_count)
                    if self._confirm_pending is not None:
                        self._release("native_target_plan")
                    self._native_leg = node.iid
                    prior = self._routing_native_goto
                    self._routing_native_goto = True
                    try:
                        return super()._execute_decider_result(observation,result,event,images)
                    finally:
                        self._routing_native_goto = prior
                        if self.mode != "nav" or self.target_instance_id != node.iid:
                            self._native_leg = None
        return super()._execute_decider_result(observation,result,event,images)

    def _pursue(self, obs, requested=None):
        if self._routing_native_goto:
            return None
        return super()._pursue(obs,requested)

    def _return(self, observation, nodes):
        if self._routing_native_goto:
            return None
        return super()._return(observation,nodes)

    def _auto_report(self, obs):
        if self._native_leg is not None and self.mode == "nav" and self.target_instance_id == self._native_leg:
            # Keep normal verified side reports possible without starting a competing motion.
            from report_agent import ReportFrontier
            return ReportFrontier._auto_report(self,obs)
        return super()._auto_report(obs)

    def _handle_arrival_transition(self, obs):
        self._native_leg = None
        entering = self._routing_native_goto
        self._routing_native_goto = False
        try:
            return super()._handle_arrival_transition(obs)
        finally:
            self._routing_native_goto = entering

    def _clear_current_target(self):
        self._native_leg = None
        return super()._clear_current_target()

    def _arrival_matches(self,node,rgb,regions):
        matches = {}
        for fid,ref in reversed(self._references(node)):
            key = (int(fid),tuple(ref.roi))
            source = self._trusted_pixels.get(key)
            if source is None:
                continue
            if key not in self._arrival_templates:
                self._arrival_templates[key] = prepare_reference(*source)
            idx = match_prepared(self._arrival_templates[key],rgb,regions)
            if idx is not None:
                matches[int(idx)] = (fid,ref)
        return matches

    def _report_gate(self, node, obs):
        ok, reason = super()._report_gate(node,obs)
        request = self._arrival_request
        if (ok or request is None or request[:2] != (int(obs.step_count),node.iid)
                or reason not in MISSING_GEOMETRY | {"need_metric_near_view"}):
            return ok,reason
        if self._known_owner(node) is not None or (getattr(self,"_goal_images",None)
                and self._instance_goal_index.get(node.iid) is None):
            return False,"arrival_identity_unavailable"
        distance = self._instance_distance_m(node)
        near = distance is not None and math.isfinite(distance) and 0 <= distance <= self.report_near_dist_m
        if not (near or (request[2]=="arrival" and request[3])):
            return False,reason
        with self._reference_purpose(False):
            rgb = _rgb(obs.rgb)
            if rgb is None:
                return False,"arrival_missing_rgb"
            boxes = eligible_boxes(self._boxes(("live",obs.step_count),rgb,self._category(obs)),rgb.shape)
            if not boxes:
                return False,"arrival_no_local_box"
            regions = np.asarray(boxes,dtype=float)*rgb.shape[1]/WORK_WIDTH
            matches = self._arrival_matches(node,rgb,regions)
            if len(matches)!=1:
                return False,"arrival_no_unique_correspondence"
            idx, (fid,ref) = next(iter(matches.items()))
            for other in self.instance_store.nodes:
                if other.iid==node.iid:
                    continue
                same = relation(self._records(node),self._records(other),
                                self._instance_goal_index.get(node.iid),self._instance_goal_index.get(other.iid))
                if same=="same":
                    if other.reported:
                        return False,"arrival_already_claimed"
                    continue
                # A rival which also fits several detections must still veto
                # the contested region. Inspect it on its own for this check.
                if self._arrival_matches(other,rgb,regions[idx:idx+1]):
                    return False,"arrival_other_identity_matches"
            self._supported_reference=(int(obs.step_count),int(fid),ref)
            if self._rival_claims(node,boxes[idx],obs,boxes):
                return False,"other_instance_explains_same_live_region"
            return True,"explicit_report_with_local_correspondence"
