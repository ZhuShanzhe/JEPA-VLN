"""Issue a short observation-route decision only when the search has a real choice."""
from collections import deque
import math


class ObservationGateway:
    def __init__(self):
        self.recent = deque(maxlen=32)

    def admit(self, *, upper_mode, accepted_cue, anchors, blocked, budget, region,
              native_priority=False, occupied=False):
        if native_priority or occupied:
            return None, "higher_priority"
        if blocked or budget < 1 or region is None:
            return None, "blocked_or_no_regional_plan"
        if upper_mode == "inspect":
            if accepted_cue and anchors:
                return "verify", "localized_visual_cue"
            return None, "visual_cue_unavailable"
        if upper_mode == "explore":
            return "discover", "inspect_observed_map"
        return None, "no_exploration_decision"

    def discoverable(self, proposal, pose, scale, budget):
        options = proposal["candidates"]
        continued = next((item for item in options if item["id"] == proposal["continue_id"]), None)
        if continued is None:
            return False, "missing_continue"
        if not continued["actions"]:
            return False, "no_comparable_continue"
        original = continued["actions"]
        eligible = []
        for item in options:
            if item is continued or not item["actions"]:
                continue
            if not any(a != b for a,b in zip(original, item["actions"])):
                continue
            if item["new_surfaces"] and item["surface_keys"]:
                eligible.append(item)
        if not eligible:
            return False, "no_distinct_unseen_surface"
        surfaces = frozenset(tuple(key) for row in eligible for key in row["surface_keys"])
        patch = (round(pose[0]*scale / .4), round(pose[1]*scale / .4),
                 round(math.degrees(pose[2]) / 30) % 12)
        for old_patch,old_surfaces,old_budget in self.recent:
            if (old_patch == patch and 0 <= old_budget-budget <= 12
                    and surfaces.issubset(old_surfaces)):
                return False, "recently_considered_surface"
        self.recent.append((patch, surfaces, budget))
        return True, "reachable_new_surface"
