"""Process pending map keyframes after a local observation maneuver has ended."""
from dataclasses import dataclass
import math
from types import SimpleNamespace

from weak_route_agent import WeakRouteAgent


@dataclass(frozen=True)
class PendingView:
    instance: int
    observations: tuple
    goal_index: object
    category: str
    frame: int
    captured_step: int
    camera_pose: tuple


class DelayedRelookAgent(WeakRouteAgent):
    def _init_report_memory(self):
        super()._init_report_memory()
        self._captured_context = {}
        self._waiting_for_map = {}
        self._last_pending_check = -1
        self._leave_after_duplicate = None

    def _capture_relook(self, observation, node, pose, scale):
        super()._capture_relook(observation, node, pose, scale)
        for fid, step in self._relook.captures.items():
            if step == observation.step_count and fid not in self._captured_context:
                self._captured_context[fid] = PendingView(
                    node.iid, tuple(node.observation_ids), self._instance_goal_index.get(node.iid),
                    self._category(observation), fid, step, tuple(map(float, pose)))

    def _end_relook(self, reason):
        state = self._relook
        if state is not None:
            for fid in state.captures:
                ctx = self._captured_context.pop(fid, None)
                if ctx and fid not in state.checked and reason in ('no_mapped_identifiable_view',
                        'motion_interrupted','updated_map_blocked','different_requested_instance'):
                    self._waiting_for_map[fid] = ctx
                    self._note('relook_awaiting_map', ctx.instance, capture_step=ctx.captured_step, frame=fid)
        super()._end_relook(reason)

    def _poll_pending_views(self, observation):
        step = int(observation.step_count)
        if self._last_pending_check == step or not self._waiting_for_map:
            return
        self._last_pending_check = step
        snapshots = None
        scale = self._metric_scale_value()
        for fid, ctx in tuple(self._waiting_for_map.items()):
            node = self.instance_store.get(ctx.instance)
            if (node is None or node.reported or tuple(node.observation_ids) != ctx.observations
                    or self._instance_goal_index.get(ctx.instance) != ctx.goal_index
                    or self._category(observation) != ctx.category or self._known_owner(node) is not None):
                reason = 'superseded_registration'
            elif step-ctx.captured_step > 40:
                reason = 'expired'
            elif self._pending_report is not None or self.align_R is None or not scale or not math.isfinite(scale):
                continue
            else:
                with self._reference_purpose(False):
                    if self._references(node):
                        self._waiting_for_map.pop(fid)
                        self._note('relook_delayed_result',ctx.instance,frame=fid,reason='trusted_reference_exists')
                        continue
                if snapshots is None:
                    snapshots = self._map_snapshot()
                if fid not in snapshots:
                    continue
                # The inherited verifier uses this context only to stamp the original
                # observation time when adding a reference. It fetches the original
                # RGB by fid and checks it against that exact point grid and pose.
                active = self._relook
                self._relook = SimpleNamespace(captures={fid:ctx.captured_step})
                try:
                    reason, owner = super()._inspect_capture(node, fid, snapshots[fid], scale)
                finally:
                    self._relook = active
                if owner is not None:
                    self._duplicate_cooldown[ctx.instance] = (ctx.observations, step+32)
                    if self._tracking(ctx.instance):
                        self._leave_after_duplicate = (ctx.instance,ctx.observations,step)
                if reason=='registered' and active is not None and active.iid==ctx.instance:
                    self._end_relook('trusted_reference_acquired')
            self._waiting_for_map.pop(fid,None)
            self._note('relook_delayed_result',ctx.instance,frame=fid,capture_step=ctx.captured_step,
                       delay_steps=step-ctx.captured_step,reason=reason)

    def _tracking(self, iid):
        return (self.target_instance_id==iid or self._confirm_pending==iid or
                (self._relook is not None and self._relook.iid==iid) or
                (self._pursuit is not None and self._pursuit.iid==iid))

    def _consume_departure(self, obs):
        if self._leave_after_duplicate is None:
            return None
        iid, signature, step = self._leave_after_duplicate
        self._leave_after_duplicate = None
        node = self.instance_store.get(iid)
        if (obs.step_count-step<=1 and node and tuple(node.observation_ids)==signature
                and self._tracking(iid) and not self._pending_report):
            if self._relook and self._relook.iid==iid:
                self._end_relook('reported_instance')
            return self._depart(obs,node)
        return None

    def _auto_report(self, obs):
        self._poll_pending_views(obs)
        action=self._consume_departure(obs)
        return action if action is not None else super()._auto_report(obs)

    def _execute_decider_result(self, obs, result, event, images=None):
        self._poll_pending_views(obs)
        if self._leave_after_duplicate:
            target=self.instance_store.get(result.target_id) if result.action in ('REPORT_FOUND','GOTO_INSTANCE') else None
            if target and target.iid==self._leave_after_duplicate[0]:
                action=self._consume_departure(obs)
                if action is not None:
                    return action
            else:
                self._leave_after_duplicate=None
        return super()._execute_decider_result(obs,result,event,images)

    def _record_and_update(self, observation, action):
        # Covers native adjustment and replay phases which bypass both policy hooks.
        # Never change an already selected action or issue a report from history.
        self._poll_pending_views(observation)
        return super()._record_and_update(observation,action)
