"""Optional native MOST exploration arm with bounded camera-target execution."""
from inspect_agent import InspectOnlyAgent
from search_agent import PriorApproachAgent
from commit_policy import CommittedInspectPolicy


class CommittedInspectAgent(InspectOnlyAgent):
    policy_class = CommittedInspectPolicy

    def _activate_frontier(self, obs, cluster, source):
        # Identical native frontier/scan installation, followed by the budget-aware
        # version of the fast geometry hook. No process-global monkeypatch.
        applied = PriorApproachAgent._activate_frontier(self, obs, cluster, source)
        scan = self.scan
        if not applied or scan is None or self.local.target is None:
            return applied
        frame = self._current_frame()
        if frame is None or self.local.target is None or self.scan is not scan:
            return applied
        pose, scale = frame
        grid = self.motion_memory.grid(self._frontier_grid, scale, self.now)
        decision = self._gated.fast_continue(self.local, grid, pose, scale, obs.max_steps-self.now,
                                             blocked=self._local_blocked)
        if decision is not None and decision["actions"]:
            self._end_scan("retained_goal_geometry_check")
            self._bind_choice(obs, scan, None, decision, None)
        return applied

    def _bind_choice(self, obs, scan, native, decision, region):
        super()._bind_choice(obs, scan, native, decision, region)
        active = self._focus_view
        if active and decision["action_source"] == "local":
            follower, tag, start, deadline = active
            if tag == f"local/{decision['goal_id']}/{decision['age']}":
                # Carrier normally gets up to eight extra correction primitives.
                # Enforce the horizon on actual environment steps as well.
                deadline = min(deadline, obs.step_count + decision["commit_actions_left"])
                self._focus_view = follower, tag, start, deadline

    def _explore_follow(self, obs):
        active = self._focus_view
        if active and active[1].startswith("local/") and self.local.target is not None:
            if self._gated._remaining(self.local, obs.max_steps-obs.step_count) <= 0:
                self._release_owned_carrier("inspection_budget")
                # Let MOST select its next action. Leave the pending target for
                # measured arrival, a higher-priority event or the next scan to close.
                return None
        return super()._explore_follow(obs)
