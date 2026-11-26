"""Plan executable moves to a visible stand-off position on the agent's map."""
import heapq
import itertools
import math

from confirm_view import FORWARD, LEFT, RIGHT, TURN, clear_move, clear_sightline, pose_after


def plan_approach(grid, pose, point, scale, near_m, blocked=(), *, visited_views=(),
                  max_actions=32, observation_view=None):
    """Return primitives, () at an admissible view, or None if none is reachable."""
    if (grid is None or scale is None or not math.isfinite(scale) or scale <= 0
            or not all(map(math.isfinite, (*pose, *point[:2], near_m))) or near_m < .32):
        return None
    if observation_view is not None and (len(observation_view) != 3 or
            not all(map(math.isfinite, observation_view))):
        return None
    def distance(a, b):
        return math.dist(a[:2], b[:2]) * scale
    def heading_error(p):
        if observation_view is not None:
            return abs(math.remainder(observation_view[2]-p[2], math.tau))
        return abs(math.remainder(math.atan2(point[1]-p[1], point[0]-p[0])-p[2], math.tau))
    def lower_bound(p):
        if observation_view is not None:
            return max(0., distance(p, observation_view)-.16) / .25
        return max(0., distance(p, point)-near_m) / .25
    def at_view(p):
        if observation_view is not None:
            return (distance(p, observation_view) <= .16 and heading_error(p) <= TURN*.52
                    and all(distance(p, v) >= .22 for v in visited_views))
        return (.32 <= distance(p, point) <= near_m and heading_error(p) <= TURN * .52
                and all(distance(p, v) >= .22 for v in visited_views)
                and clear_sightline(grid, p, point[:2], scale))
    serial = itertools.count()
    queue = [(lower_bound(pose), next(serial), tuple(pose), ())]
    costs = {}
    while queue and len(costs) < 9000:
        _, _, current, actions = heapq.heappop(queue)
        key = (round((current[0]-pose[0])*scale/.085), round((current[1]-pose[1])*scale/.085),
               round((current[2]-pose[2])/TURN) % 12)
        if costs.get(key, max_actions+1) <= len(actions):
            continue
        costs[key] = len(actions)
        if at_view(current):
            return actions
        if len(actions) >= max_actions:
            continue
        for action in (FORWARD, LEFT, RIGHT):
            if actions and (actions[-1], action) in ((LEFT, RIGHT), (RIGHT, LEFT)):
                continue
            if action == FORWARD and any(distance(current, p) < .16 and
                    abs(math.remainder(current[2]-p[2], math.tau)) < .75*TURN for p in blocked):
                continue
            dest = pose_after(current, action, scale)
            if action == FORWARD and not clear_move(grid, current, dest):
                continue
            new_cost = len(actions) + 1
            if new_cost + lower_bound(dest) > max_actions + 1e-5:
                continue
            priority = new_cost + 1.35*lower_bound(dest) + .12*heading_error(dest)/TURN
            heapq.heappush(queue, (priority, next(serial), dest, actions + (action,)))
    return None
