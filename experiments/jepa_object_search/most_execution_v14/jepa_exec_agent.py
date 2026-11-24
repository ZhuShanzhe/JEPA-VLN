"""MOST-Nav agent entry point: jepa_exec_agent:JepaExecutionFrontier."""
from dataclasses import dataclass
import math
from numbers import Integral

from benchmark_api import Action
from jepa_pilot_agent import JepaFrontier


@dataclass
class PendingBranch:
    carrier: object
    native_next: object
    start: int
    original: str
    selected: str
    prefix: tuple
    budget: int
    cursor: int = 0
    sent_at: int = -1
    hook: object = None
    expected_pose: tuple = ()
    anchor: int = -1


class JepaExecutionFrontier(JepaFrontier):
    """Carry an accepted choice through the first different primitive and yield."""
    commit = None

    def _close_commit(self, reason):
        r = self.commit
        if r is None:
            return
        if r.carrier.next_action is r.hook:
            r.carrier.next_action = r.native_next
        self.commit = None
        self.emit("commit_end", reason=reason, start=r.start, original=r.original,
                  selected=r.selected, confirmed_steps=r.cursor, fork_offset=len(r.prefix)-1,
                  outstanding_step=None if r.sent_at < 0 else r.sent_at)

    def accept_selection(self, observation, original_id, selected_id, original_plan, selected_plan):
        """Bind the *already activated* frontier to a bounded action prefix."""
        self._close_commit("superseded_selection")
        if selected_id == original_id:
            return False
        for plan in (original_plan, selected_plan):
            if any(isinstance(a, bool) or not isinstance(a, Integral) or a not in (1, 2, 3)
                   for a in plan):
                raise ValueError("invalid navigation action in accepted frontier plan")
        fork = next((i for i, (a, b) in enumerate(zip(original_plan, selected_plan))
                     if a != b), None)
        fl = self._explore_follower
        remaining = int(observation.max_steps) - self.now
        if fork is None or fl is None or fork + 1 > remaining:
            reason = ("no_distinct_command" if fork is None else
                      "no_active_follower" if fl is None else "insufficient_steps")
            self.emit("commit_bypass", reason=reason, original=original_id, selected=selected_id)
            return False
        # The candidate paths are in the map frame. A newly activated native
        # follower starts at (0, 0, 0) until it receives its first mapper anchor.
        pose = getattr(self, "_frontier_pose", None)
        if pose is None:
            pose = getattr(self, "_frontier_slam_pose", None)
        if fl.anchor_frame < 0 and pose is not None:
            fl.x, fl.y, fl.yaw = map(float, pose)
        r = PendingBranch(fl, fl.next_action, self.now, original_id, selected_id,
                          tuple(map(int, selected_plan[:fork+1])), int(observation.max_steps),
                          expected_pose=(fl.x, fl.y, fl.yaw), anchor=fl.anchor_frame)
        r.hook = lambda: self._dispatch(r)
        self.commit = r
        fl.next_action = r.hook
        self.emit("commit_start", start=r.start, selected=selected_id, original=original_id,
                  prefix=list(r.prefix), fork_offset=fork)
        return True

    def _dispatch(self, r):
        fl = r.carrier
        # Run the native waypoint bookkeeping even when committing a different
        # primitive. A lost path, arrival or material relocalization always wins.
        native, arrived = r.native_next()
        if fl.anchor_frame != r.anchor:
            x, y, yaw = r.expected_pose
            distance = math.hypot(fl.x-x, fl.y-y) * fl.scale
            turn = abs(math.remainder(fl.yaw-yaw, 2*math.pi))
            if distance > .25 or turn > math.radians(20):
                self._close_commit("map_pose_corrected")
                return native, arrived
        if arrived or native is None or self.now >= r.budget:
            self._close_commit("native_path_ended" if arrived or native is None else "budget_exhausted")
            return native, arrived
        if r.sent_at >= 0:
            raise RuntimeError("second frontier dispatch before observation acknowledgement")
        action = r.prefix[r.cursor]
        r.sent_at = self.now
        self.emit("commit_dispatch", start=r.start, offset=r.cursor, action=action,
                  native_suggestion=int(native), selected=r.selected)
        return action, False

    def _explore_follow(self, observation):
        r = self.commit
        if r and self._explore_follower is not r.carrier:
            self._close_commit("native_follower_replaced")
        action = super()._explore_follow(observation)
        r = self.commit
        if r and r.sent_at == self.now:
            # Native _explore_follow has applied the returned action to odometry.
            fl = r.carrier
            r.expected_pose, r.anchor = (fl.x, fl.y, fl.yaw), fl.anchor_frame
        return action

    def _rank_after_scan(self, observation, scan):
        super()._rank_after_scan(observation, scan)
        ranked = self.route
        if ranked is None:
            return
        # Retire the legacy passive matcher. Only a dispatched primitive with
        # a matching subsequent observation belongs to the execution receipt.
        self.route = None
        self.accept_selection(observation, ranked["original"], ranked["selected"],
                              scan["plans"][0], ranked["plan"])

    def _feed_frame(self, observation):
        old_step = self.observed_step
        result = super()._feed_frame(observation)
        r = self.commit
        if r is None or self.now == old_step or self.now <= r.start:
            return result
        if r.sent_at != self.now-1 or observation.previous_action != r.prefix[r.cursor]:
            self._close_commit("command_not_acknowledged")
        elif r.prefix[r.cursor] == int(Action.MOVE_FORWARD) and self._last_motion_failed:
            self._close_commit("forward_blocked")
        else:
            r.cursor += 1
            r.sent_at = -1
            if r.cursor == len(r.prefix):
                self._close_commit("first_difference_observed")
        return result

    def _activate_frontier(self, observation, cluster, source):
        self._close_commit("native_reactivation")
        return super()._activate_frontier(observation, cluster, source)

    def _execute_decider_result(self, observation, result, event, images=None):
        self._close_commit("native_decision_" + str(result.action))
        return super()._execute_decider_result(observation, result, event, images)

    def init_episode(self, observation):
        self._close_commit("episode_reset")
        super().init_episode(observation)

    def act(self, observation):
        action = super().act(observation)
        r = self.commit
        if r:
            if self._explore_follower is not r.carrier:
                self._close_commit("native_follower_replaced")
            elif r.sent_at != self.now or int(action) != r.prefix[r.cursor]:
                self._close_commit("higher_priority_action")
            elif self.now + 1 == int(observation.max_steps):
                self.emit("commit_waiting_final_observation", start=r.start, action=int(action))
        return action
