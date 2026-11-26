"""A trusted sighting can initiate a bounded, interruptible approach to a report view."""
import math
from dataclasses import dataclass, field

from agents import navigator
from approach_motion import plan_approach
from barrier_agent import BarrierAgent
from confirm_view import FORWARD, LEFT, RIGHT, TURN, clear_move, pose_after
from report_evidence import eligible_boxes
from trusted_reference import TrustedReferenceMixin
from visual_witness import VisualWitnessMixin


@dataclass
class Pursuit:
    iid: int
    first_step: int
    actions: tuple = ()
    point: tuple = ()
    expected: tuple = ()
    origin: tuple = ()
    sent: int = -1
    last: int = -1
    blocked: list = field(default_factory=list)
    inspected: list = field(default_factory=list)
    tried_ranked: bool = False
    offered_view_step: int = -1
    witness_pose: tuple = ()
    witness_turn: int = -1
    attempted_return: bool = False
    ranked_leg: bool = False


class ApproachMixin:
    def _init_report_memory(self):
        super()._init_report_memory()
        self._pursuit = None
        self._paused_sightings = {}
        self._approach_attempts = {}
        self._pursuit_reply = None

    def _release_pursuit(self, step, reason, pause=True):
        state = self._pursuit
        if state is not None:
            if pause:
                self._paused_sightings[state.iid] = step + 24
            self._note("approach_end", state.iid, reason=reason, elapsed=step-state.first_step)
        self._pursuit = None
        self._pursuit_reply = None

    def _usable_sighting(self, node, step):
        if (node is None or self._known_owner(node) is not None
                or (getattr(self, "_goal_images", None) and self._instance_goal_index.get(node.iid) is None)
                or step < self._paused_sightings.get(node.iid, -1)):
            return False
        with self._reference_purpose(False):
            return bool(self._references(node))

    def _remembered_views(self, node, scale):
        if self.align_R is None:
            return
        with self._reference_purpose(False):
            refs = tuple(reversed(self._references(node)))
        for fid, ref in refs:
            camera = self.client.get_frame_pose(fid)
            if camera is None:
                continue
            view = tuple(navigator.pose_to_yaw_2d(camera, self.align_R))
            if (all(map(math.isfinite, view)) and math.dist(view[:2],node.point[:2])*scale <= 2.25):
                yield fid, ref, view

    def _report_gate(self, node, obs):
        allowed, reason = super()._report_gate(node,obs)
        if allowed or reason != "need_metric_near_view":
            return allowed, reason
        pose, scale = self._estimated_current_pose(), self._metric_scale_value()
        if pose is None or scale is None or not math.isfinite(scale) or scale <= 0:
            return False, reason
        # A known, previously occupied viewing position is a separate range
        # proxy from distance to an object's estimated center. It does not
        # identify a benchmark viewpoint. Require an independent live match.
        boxes = None
        with self._reference_purpose(False):
            for fid, ref, view in self._remembered_views(node, scale):
                if (math.dist(pose[:2],view[:2])*scale > .16 or
                        abs(math.remainder(pose[2]-view[2],math.tau)) > TURN*.55):
                    continue
                if boxes is None:
                    boxes = eligible_boxes(self._boxes(("live",obs.step_count),obs.rgb,self._category(obs)),
                                           obs.rgb.shape)
                if not boxes:
                    return False, "no_local_target_box"
                match, verdict = self._support(fid, ref, obs, boxes)
                if match is not None:
                    if self._rival_claims(node,match,obs,boxes):
                        return False,"other_instance_explains_same_live_region"
                    return True,"live_identity_at_remembered_camera"
                reason = verdict
        return False, reason

    def _pursue(self, obs, requested=None):
        step = int(obs.step_count)
        if self._pending_report:
            if self._pending_report.step == step:
                from benchmark_api import Action
                return int(Action.TARGET_FOUND)
            return None
        quota = getattr(obs, "target_count", 0)
        if quota and self._reported_count >= quota:
            self._release_pursuit(step, "all_requested_reports_dispatched", pause=False)
            return None
        if self._pursuit_reply and self._pursuit_reply[0] == step:
            return self._pursuit_reply[1]
        self._collect_views()
        state = self._pursuit
        pose = self._estimated_current_pose()
        scale = self._metric_scale_value()
        grid = self._frontier_grid
        if (pose is None or scale is None or not math.isfinite(scale) or scale <= 0
                or not all(map(math.isfinite, pose)) or grid is None):
            self._release_pursuit(step, "map_or_pose_unavailable", pause=False)
            return None
        if state is not None:
            node = self.instance_store.get(state.iid)
            if not self._usable_sighting(node, step):
                self._release_pursuit(step, "sighting_resolved_or_no_longer_trusted", pause=False)
                state = None
        if state is None:
            nodes = [requested] if requested is not None else self.instance_store.available()
            choices = []
            for candidate in nodes:
                if not self._usable_sighting(candidate, step):
                    continue
                signature = (candidate.iid, tuple(candidate.observation_ids))
                if self._approach_attempts.get(signature, 0) >= 2:
                    continue
                distance = self._instance_distance_m(candidate)
                if distance is not None and math.isfinite(distance) and distance >= 0.:
                    choices.append((candidate.iid != self.target_instance_id, distance, candidate.iid, candidate))
            if not choices:
                return None
            node = min(choices, key=lambda item: item[:3])[-1]
            signature = (node.iid, tuple(node.observation_ids))
            self._approach_attempts[signature] = self._approach_attempts.get(signature, 0) + 1
            state = self._pursuit = Pursuit(node.iid, step)
            if self._confirm_view_state:
                self._end_confirm_view("registered_sighting_priority", step)
            self._note("approach_start", node.iid, distance_m=float(self._instance_distance_m(node)))

        # The full inherited gate owns range, trusted reference, rival claims and report dispatch.
        allowed, reason = self._report_gate(node, obs)
        if allowed:
            action = self._stage_report(node, obs, reason)
            self._release_pursuit(step, "report_dispatched", pause=False)
            return action
        if reason in ("other_instance_explains_same_live_region", "identity_different"):
            self._release_pursuit(step, reason)
            return None
        remaining = min(32-(step-state.first_step), getattr(obs,"max_steps",step+33)-step-1)
        if remaining < 1:
            self._release_pursuit(step,"approach_budget_spent")
            return None
        pose = tuple(map(float, pose))
        point = tuple(map(float, node.point[:2]))
        if not all(map(math.isfinite, point)):
            self._release_pursuit(step,"unavailable_target_location")
            return None
        if state.expected:
            received = step == state.last+1 and obs.previous_action == state.sent
            stalled = received and state.sent == FORWARD and getattr(self,"_last_motion_failed",False)
            if stalled:
                state.blocked.append(state.origin)
            if (not received or stalled or math.dist(pose[:2],state.expected[:2])*scale > .20
                    or abs(math.remainder(pose[2]-state.expected[2],math.tau)) > TURN*.6
                    or math.dist(point,state.point)*scale > .20):
                state.actions = ()
        near = min(.65, float(self.report_near_dist_m))
        if state.actions and state.actions[0] == FORWARD and not clear_move(
                grid, pose, pose_after(pose, FORWARD, scale)):
            state.actions = ()
        # Preserve the position and bearing of the first credible live
        # observation, even when the current movement sequence is unfinished.
        if state.offered_view_step < state.first_step:
            with self._reference_purpose(False):
                refs = self._references(node)
            for fid, ref in reversed(refs):
                witness = self._visual_history.get((int(fid),tuple(ref.roi)))
                if witness is not None and witness[0] == step and not witness[-1]:
                    from appearance import WORK_WIDTH
                    x, _, u, _ = witness[2]
                    state.witness_turn = LEFT if (x+u)/2 < WORK_WIDTH/2 else RIGHT
                    state.witness_pose = pose
                    if not (state.ranked_leg and state.actions):
                        state.actions = (state.witness_turn,)
                    state.offered_view_step = step
                    self._note("approach_reobserve",node.iid,reason="one_live_visual_witness")
                    break
        elif (state.witness_pose and not state.attempted_return and
              step-state.offered_view_step in (1,2) and not (state.ranked_leg and state.actions)):
            # One bounded return to the actual sighting. A small angle change
            # is enough to acquire a second view without driving into the wall.
            back = plan_approach(grid,pose,point,scale,near,state.blocked,max_actions=min(remaining,8),
                                 observation_view=state.witness_pose)
            state.attempted_return = True
            if back:
                state.actions = back
                self._note("approach_recover_sighting",node.iid,actions=len(back))
        if not state.actions:
            def plan(view=None):
                return plan_approach(grid, pose, point, scale, near, state.blocked,
                                     visited_views=state.inspected,max_actions=remaining,observation_view=view)
            actions = None
            destination = "object_standoff"
            if not state.tried_ranked and math.dist(pose[:2],point)*scale <= 1.8:
                state.tried_ranked = True
                with self._reference_purpose(False):
                    refs = self._references(node)
                if refs:
                    fid, ref = refs[-1]
                    ranked = self._rank_probe(fid,ref.roi,pose,point,scale,state.inspected,min(12,remaining))
                    if ranked and 0 < len(ranked["sequence"]) <= min(12,remaining):
                        sequence = tuple(ranked["sequence"])
                        if (all(a in (LEFT,RIGHT,FORWARD) for a in sequence)
                                and (sequence[0] != FORWARD or clear_move(
                                    grid,pose,pose_after(pose,FORWARD,scale)))):
                            actions = sequence
                            destination = "learned_verification_view"
            for _, _, view in self._remembered_views(node,scale) if actions is None else ():
                if any(math.dist(v[:2],view[:2])*scale < .38 for v in state.inspected):
                    continue
                actions = plan(view)
                if actions is not None:
                    destination = "remembered_camera"
                    break
            if actions is None:
                actions = plan()
            if actions == ():
                # The estimated pose is close but identity is unresolved. Acquire
                # fresh evidence from another reachable camera, once per position.
                state.inspected.append(pose[:2])
                actions = plan() if len(state.inspected) <= 3 else None
                destination = "alternative_visible_position"
            if not actions:
                self._release_pursuit(step,"no_fresh_reachable_view")
                return None
            state.actions = actions
            state.ranked_leg = destination == "learned_verification_view"
            self._note("approach_plan",node.iid,actions=len(actions),prior_verdict=reason,view=destination)
        action, *tail = state.actions
        state.actions = tuple(tail)
        state.origin, state.expected = pose, pose_after(pose, action, scale)
        state.last, state.sent, state.point = step, action, point
        self._pursuit_reply = (step, action)
        self._view_last_reply = (step, node.iid, (action,"confirm_view_motion"))
        return action

    def _auto_report(self, obs):
        action = self._pursue(obs)
        return action if action is not None else super()._auto_report(obs)

    def _recover_report(self, node, obs):
        action = self._pursue(obs, node)
        return (action, "approach_registered_sighting") if action is not None else super()._recover_report(node, obs)

    def _execute_decider_result(self, observation, result, event, images=None):
        if result.action in {"EXPLORE","GOTO_FRONTIER","GOTO_INSTANCE","FINISH"}:
            quota = getattr(observation,"target_count",0)
            if result.action != "FINISH" or (quota and self._reported_count < quota):
                target = self.instance_store.get(result.target_id) if result.action == "GOTO_INSTANCE" else None
                action = self._pursue(observation, target)
                if action is not None:
                    return action
        return super()._execute_decider_result(observation, result, event, images)

    def _build_decider_input(self, observation, local_map=False):
        state, image = super()._build_decider_input(observation, local_map)
        current = self._pursuit
        if current is not None:
            state = dict(state, active_instance_approach={
                "instance_id": current.iid, "elapsed_actions": int(observation.step_count)-current.first_step,
                "purpose": "Acquire a close, unambiguous view and report the registered instance."})
        return state, image


class ApproachAgent(ApproachMixin, TrustedReferenceMixin, VisualWitnessMixin, BarrierAgent):
    """MOST-Nav actions and sensors with an active approach and local visual confirmation."""
