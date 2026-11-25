"""Current-view report verification and bounded recovery for the native action loop."""
import math

from benchmark_api import Action
from report_evidence import current_support, eligible_boxes, overlap


class ReportRecoveryMixin:
    def _init_report_memory(self):
        super()._init_report_memory()
        self._report_feedback = {}
        self._live_verdict_cache = {}
        self._verdict_step = None

    def _support(self, fid, ref, observation, boxes):
        if self._verdict_step != observation.step_count:
            self._live_verdict_cache.clear()
            self._verdict_step = observation.step_count
        key = (fid, id(ref))
        if key not in self._live_verdict_cache:
            match, _ = self._live_match(fid, ref, observation)
            self._live_verdict_cache[key] = current_support(ref, match, observation.rgb, boxes)
        return self._live_verdict_cache[key]

    def _report_gate(self, node, observation):
        self._collect_views()
        refs = self._references(node)
        if not refs:
            return False, "need_grounded_object_region"
        reason = "need_metric_near_view"
        boxes = None
        for fid, ref in reversed(refs):
            geo = self._geometry(node, fid)
            if geo is None or geo[2] > self.report_near_dist_m:
                continue
            same_view = geo[0] <= .35 and geo[1] <= self.report_near_dist_m and abs(geo[3]) <= math.radians(30)
            if geo[2] > min(.70, self.report_near_dist_m) and not same_view:
                continue
            if boxes is None:
                rows = self._boxes(("live", observation.step_count), observation.rgb, self._category(observation))
                boxes = eligible_boxes(rows, observation.rgb.shape)
            if not boxes:
                return False, "no_local_target_box"
            box, reason = self._support(fid,ref,observation,boxes)
            if box is None:
                continue
            if self._rival_claims(node,box,observation,boxes):
                reason = "other_instance_explains_same_live_region"
                continue
            return True, reason
        return False, reason

    def _rival_claims(self, node, box, obs, boxes):
        # Keep the native global ambiguity checks. Also apply the added
        # localized check to every nearby rival, including executed reports.
        from identity import relation
        if self._competing_identity(node,box,obs,False):
            return True
        for other in self.instance_store.nodes:
            if other.iid==node.iid:
                continue
            distance=self._instance_distance_m(other)
            if distance is not None and distance>self.report_near_dist_m+1.5:
                continue
            if relation(self._records(node),self._records(other),self._instance_goal_index.get(node.iid),
                        self._instance_goal_index.get(other.iid))=="same":
                continue
            for fid,ref in self._references(other):
                rival,_=self._support(fid,ref,obs,boxes)
                if rival is not None and overlap(rival,box)>=.30:
                    return True
        return False

    def _build_decider_input(self, observation, local_map=False):
        state, image = super()._build_decider_input(observation,local_map)
        recent=[]
        for iid,entry in self._report_feedback.items():
            node=self.instance_store.get(iid)
            if node is not None and not node.reported and observation.step_count-entry["step"]<=24:
                recent.append(dict(instance_id=iid,**entry))
        if recent:
            state=dict(state,report_recovery={
                "recent":recent[-4:],
                "guidance":"Reports need a live local detection matching the registered instance at close range. "
                           "Continue the recovery route, inspect a different reachable view, or pursue another "
                           "instance. Repeat a report only after obtaining fresh close visual evidence."})
        return state,image

    def _report_found(self,instance_id=None):
        node=self.instance_store.get(instance_id)
        if node is None or self._pending_report is not None or self._known_owner(node) is not None:
            return super()._report_found(instance_id)
        obs=self._last_observation
        self._requested.add(node.iid)
        ok,reason=self._report_gate(node,obs)
        if ok:
            self._report_feedback.pop(node.iid,None)
            return self._stage_report(node,obs,reason)
        self._note("report_deferred",node.iid,reason=reason)
        self._stamp_pending_outcome("deferred:"+reason,"Keep this instance open. Move to a fresh close view or inspect another frontier.")
        action, plan=self._recover_report(node,obs)
        self._report_feedback[node.iid]=dict(step=int(obs.step_count),reason=reason,recovery=plan)
        self._note("report_recovery_action",node.iid,reason=reason,recovery=plan,action=int(action))
        return action

    def _recover_report(self,node,obs):
        # Continue a running return (including a different instance) before
        # scheduling any new waypoint. Never re-enter the VLM on arrival here.
        if self.mode=="nav" and self.follower is not None:
            action,arrived,stuck=self._nav_action(obs)
            if action is not None and not arrived and not stuck:
                return action,"follow_active_path"
            self._clear_current_target()
            self.mode="explore"
        if self._explore_follower is not None and self._report_feedback.get(node.iid):
            action=self._explore_follow(obs)
            if action is not None:
                return action,"follow_fresh_frontier"
        options=[]
        for fid,_ in self._references(node):
            geo=self._geometry(node,fid)
            count,when=self._attempts.get((node.iid,fid),(0,-100))
            if geo and geo[0]>.35 and geo[1]<=2.25 and count<2 and self.now-when>=24:
                options.append((geo[1]+.22*geo[0],fid,count))
        for _,fid,count in sorted(options):
            self._attempts[node.iid,fid]=(count+1,self.now)
            if not self._schedule_revisit(fid).get("navigating"):
                continue
            self._close_commit("report_recovery")
            self._end_scan("report_recovery")
            self._end_route("report_recovery")
            self._cancel_delay("report_recovery")
            self._confirm_pending=node.iid
            self._confirm_frame=fid
            self._confirm_orient=0
            self.target_instance_id=node.iid
            action,arrived,stuck=self._nav_action(obs)
            if action is not None and not arrived and not stuck:
                return action,"return_to_registered_view"
            self._clear_current_target()
            self.mode="explore"
        # The native frontier planner issues a real collision-aware step and
        # keeps its follower for the next observation. The clue remains open.
        return self._autonomous_explore_action(obs),"choose_fresh_frontier"
