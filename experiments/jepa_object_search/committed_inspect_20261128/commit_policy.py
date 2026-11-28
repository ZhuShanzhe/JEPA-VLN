"""Let a GPT-authorized local observation finish under bounded geometric control."""
from copy import copy
import math

from local_runtime import fast_continue as geometric_continue
from trigger_policy import InspectOnlyPolicy


class CommittedInspectPolicy(InspectOnlyPolicy):
    # Same action horizon already used to propose reachable sites in the frozen arm.
    horizon = 24

    def __init__(self, gate, instruction=None):
        super().__init__(gate, instruction)
        self.goal_id = None
        self.start_remaining = None

    def _remaining(self, tracker, budget):
        if self.goal_id != tracker.target["goal_id"] or self.start_remaining is None:
            raise RuntimeError("active inspection has no matching execution grant")
        spent = self.start_remaining - budget
        if spent < 0:
            raise RuntimeError("episode action budget increased during inspection")
        return max(0, self.horizon - spent)

    def _annotate(self, decision, tracker, budget):
        if decision["action_source"] != "local":
            return
        gid = decision["goal_id"]
        if decision["event"] == "acquired":
            self.goal_id = gid
            self.start_remaining = budget
        if gid != tracker.target["goal_id"]:
            raise RuntimeError("dispatched observation site differs from active target")
        left = min(self._remaining(tracker, budget), budget)
        if left <= 0:
            raise RuntimeError("out-of-budget local dispatch")
        decision["actions"] = decision["actions"][:left]
        decision.update(commit_actions_left=left, commit_horizon=self.horizon,
                        commit_elapsed=self.start_remaining-budget)

    def fast_continue(self, tracker, grid, pose, scale, budget, blocked=False):
        if tracker.target is not None and self._remaining(tracker, budget) <= 0:
            # Fall through to a scan with regional candidates and process the expiry there.
            return None
        decision = geometric_continue(tracker, grid, pose, scale, budget, blocked)
        if decision is not None:
            self._annotate(decision, tracker, budget)
        return decision

    def pick(self, tracker, memory, grid, pose, scale, rows, views, goal, budget, channel,
             depth_m=None, hfov=90., blocked=False, replay_memory=False, audit_sites=None):
        args = (memory, grid, pose, scale, rows, views, goal, budget, channel, depth_m,
                hfov, blocked, replay_memory, audit_sites)
        if tracker.target is None:
            # Identical gate input, regional ranking, JEPA scoring and first acquisition.
            decision, region = super().pick(tracker, *args)
            self._annotate(decision, tracker, budget)
            return decision, region

        # The frozen selector can discard its target before it calls decide(). Give its
        # gate a snapshot so that a different suggestion is observed but cannot erase
        # the executor's active target. The real tracker alone owns all state transitions.
        previous = tracker.target
        advisory = copy(tracker)
        physical_release = None
        left = self._remaining(tracker, budget)

        def execute(metric, region, sites, plan, *, region_entered, blocked):
            nonlocal physical_release
            if left > 0:
                # Preserves original collision/stall checks and stronger-new-support
                # threshold for any actually offered and scored alternative site.
                result = tracker.decide(metric, region, sites, plan, region_entered=region_entered,
                                        blocked=blocked)
            else:
                tracker.tick += 1
                physical_release = tracker.arrival(metric, plan)
                if physical_release is None:
                    tracker._remember(previous, "inspection_budget")
                    physical_release = dict(goal_id=previous["goal_id"], event="inspection_budget",
                                            age=tracker.age, elapsed=self.start_remaining-budget,
                                            distance_m=math.dist(metric[:2], previous["site_xy"]))
                    tracker.target = None
                result = tracker._region_move(region, "region_after_inspection", physical_release)
            physical_release = result["released"]
            return result

        advisory.decide = execute
        decision, region = super().pick(advisory, *args)
        tracker.scan_surfaces = advisory.scan_surfaces
        tracker.may_skip_scan = advisory.may_skip_scan
        # Parent reports an advisory cancellation if it finds no physical release.
        decision["released"] = physical_release
        decision.update(commit_previous_id=previous["goal_id"], commit_before_dispatch=left,
                        advisory_status=decision["exec_status"])
        if decision["action_source"] == "local" and decision["event"] == "retained":
            decision["exec_status"] = "committed_view_retained"
        elif physical_release and physical_release["event"] == "inspection_budget":
            decision["exec_status"] = "inspection_budget"
        self._annotate(decision, tracker, budget)
        return decision, region
