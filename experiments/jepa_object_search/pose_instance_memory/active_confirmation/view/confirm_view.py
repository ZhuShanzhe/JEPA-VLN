"""Approach a close registered object and inspect it with the existing live report gate."""
import math
from collections import deque
from heapq import heappop, heappush
from itertools import count

from agents import navigator
from benchmark_api import Action


FORWARD = int(Action.MOVE_FORWARD)
LEFT = int(Action.TURN_LEFT)
RIGHT = int(Action.TURN_RIGHT)
TURN = math.pi / 6
MAX_ACTIONS = 12
MAX_ATTEMPTS = 2


def pose_after(pose, action, scale):
    x, y, yaw = pose
    if action == FORWARD:
        return x + math.cos(yaw)*.25/scale, y + math.sin(yaw)*.25/scale, yaw
    return x, y, yaw + (TURN if action == LEFT else -TURN)


def clear_move(grid, start, end):
    distance = math.dist(start[:2], end[:2])
    n = max(1, math.ceil(distance / (.5*grid.res)))
    for i in range(1, n+1):
        xy = tuple(start[j] + (end[j]-start[j])*i/n for j in (0, 1))
        cell = grid.world_to_cell(xy)
        if not grid.in_bounds(cell) or not grid.traversable(cell):
            return False
    return True


def clear_sightline(grid, pose, point, scale):
    distance = math.dist(pose[:2], point)
    # The target's own footprint and its wall can occupy the terminal cells.
    length = distance - .20/scale
    if length <= 0:
        return True
    n = max(1, math.ceil(length / (.5 * grid.res)))
    for i in range(n+1):
        ratio = length*i/(n*distance)
        cell = grid.world_to_cell(tuple(pose[j]+(point[j]-pose[j])*ratio for j in (0, 1)))
        if not grid.in_bounds(cell) or grid.raw_obstacle[cell[1], cell[0]]:
            return False
    return True


def return_to_reference(grid, start, point, reference, scale, excluded, report_range):
    """Find executable primitives reproducing the registered camera position and yaw."""
    if (reference is None or math.dist(reference[:2], point)*scale > report_range or
            math.dist(reference[:2], start[:2])*scale > 1.50 or
            any(math.dist(reference[:2], p)*scale < .25 for p in excluded)):
        return None
    def position_error(pose):
        return math.dist(pose[:2], reference[:2])*scale
    def yaw_error(pose):
        return abs(math.remainder(pose[2]-reference[2], 2*math.pi))
    if position_error(start) <= .14 and yaw_error(start) <= math.radians(15):
        return None
    serial = count()
    queue = [(0., next(serial), start, ())]
    visited = {}
    closest = None
    while queue and len(visited) < 1800:
        _, _, pose, commands = heappop(queue)
        key = (round((pose[0]-start[0])*scale/.09), round((pose[1]-start[1])*scale/.09),
               round((pose[2]-start[2])/TURN) % 12)
        if visited.get(key, MAX_ACTIONS+1) <= len(commands):
            continue
        visited[key] = len(commands)
        if (position_error(pose) <= .14 and yaw_error(pose) <= math.radians(15) and commands
                and clear_sightline(grid, pose, point, scale)):
            # Reproduce the original crop where the discrete dynamics allow it.
            # A small lateral displacement near a wall object changes its image
            # markedly. Preserve the best coarse arrival for imperfect odometry.
            score = position_error(pose) + .20*yaw_error(pose)
            if closest is None or (score, len(commands)) < closest[:2]:
                closest = score, len(commands), pose, commands
            if position_error(pose) <= .045 and yaw_error(pose) <= math.radians(5):
                return pose, commands
        if len(commands) >= MAX_ACTIONS:
            continue
        for action in (FORWARD, LEFT, RIGHT):
            if commands and (commands[-1],action) in ((LEFT, RIGHT), (RIGHT, LEFT)):
                continue
            nxt = pose_after(pose, action, scale)
            if action == FORWARD and (position_error(nxt) > 1.80 or not clear_move(grid, pose, nxt)):
                continue
            remaining = position_error(nxt)/.25
            if remaining-.14/.25 > MAX_ACTIONS-len(commands)-1:
                continue
            # While away from the reference, face the route; on arrival face
            # the *saved camera yaw*, which preserves an off-centre crop.
            desired = (math.atan2(reference[1]-nxt[1], reference[0]-nxt[0])
                       if position_error(nxt)>.045 else reference[2])
            turn_error = abs(math.remainder(desired-nxt[2],2*math.pi))/TURN
            priority = len(commands)+1 + 1.65*remaining + .8*turn_error
            heappush(queue,(priority,next(serial),nxt,commands+(action,)))
    return closest[2:] if closest is not None else None


def choose_view(grid, start, point, reference, scale, excluded, near_limit=.70):
    """Enumerate executable short action sequences on the agent's current map."""
    if (grid is None or scale is None or not math.isfinite(scale) or scale <= 0 or
            not all(map(math.isfinite, (*start, *point)))):
        return None
    if math.dist(start[:2], point)*scale > 1.80:
        return None
    queue = deque([(start, ())])
    seen = set()
    best = None
    # Quantize relative to the current pose. Headings differ by exactly 30°.
    while queue and len(seen) < 2600:
        pose, commands = queue.popleft()
        x, y, yaw = pose
        key = (round((x-start[0])*scale/.10), round((y-start[1])*scale/.10),
               round((yaw-start[2])/TURN) % 12)
        if key in seen:
            continue
        seen.add(key)
        dist = math.dist(pose[:2], point)*scale
        bearing = math.atan2(point[1]-y, point[0]-x)
        error = abs(math.remainder(bearing-yaw, 2*math.pi))
        if (commands and .32 <= dist <= near_limit and error <= math.radians(16) and
                (math.dist(pose[:2], start[:2])*scale >= .17 or
                 abs(math.remainder(yaw-start[2], 2*math.pi)) > math.radians(20)) and
                all(math.dist(pose[:2], p)*scale >= .22 for p in excluded) and
                clear_sightline(grid, pose, point, scale)):
            reference_error = 0.
            if reference is not None:
                registered = math.atan2(point[1]-reference[1], point[0]-reference[0])
                reference_error = abs(math.remainder(bearing-registered, 2*math.pi))
            cost = len(commands)*.10 + abs(dist-.54)*1.8 + error*.25 + reference_error*.20
            if best is None or cost < best[0]:
                best = cost, pose, commands
        if len(commands) >= MAX_ACTIONS or (best and (len(commands)+1)*.10 >= best[0]):
            continue
        for action in (FORWARD, LEFT, RIGHT):
            if commands and (commands[-1], action) in ((LEFT, RIGHT), (RIGHT, LEFT)):
                continue
            nxt = pose_after(pose, action, scale)
            if action == FORWARD and (math.dist(start[:2], nxt[:2])*scale > 2.0 or
                                      not clear_move(grid, pose, nxt)):
                continue
            queue.append((nxt, commands+(action,)))
    return best[1:] if best is not None else None


class ConfirmViewMixin:
    def _init_report_memory(self):
        super()._init_report_memory()
        self._confirm_view_state = None
        self._view_history = {}
        self._view_last_reply = None
        self._auto_view_last = None

    def _end_confirm_view(self, reason, step):
        state = self._confirm_view_state
        if state is None:
            return
        history = self._view_history[state["node_id"]]
        history["end"] = step
        history["excluded"].append(state["waypoint"][:2])
        self._note("confirm_view_end", state["node_id"], reason=reason, frame=state["fid"],
                   start=state["start"], commands=state["cursor"], waypoint=list(state["waypoint"]))
        self._confirm_view_state = None

    def _begin_confirm_view(self, node, obs, pose, scale):
        step = int(obs.step_count)
        hist = self._view_history.setdefault(node.iid, dict(attempts=0, end=-100, excluded=[]))
        if hist["attempts"] >= MAX_ATTEMPTS or step-hist["end"] < 12:
            return False
        refs = self._references(node)
        if not refs:
            return False
        fid = int(refs[-1][0])
        if self.align_R is None:
            return False
        camera = self.client.get_frame_pose(fid)
        reference = navigator.pose_to_yaw_2d(camera, self.align_R) if camera is not None else None
        if reference is not None and not all(map(math.isfinite, reference)):
            reference = None
        point = tuple(node.point[:2])
        if math.dist(point,pose[:2])*scale > 1.80 or not all(map(math.isfinite,point)):
            return False
        result = return_to_reference(self._frontier_grid, pose, point, reference, scale,
                                     hist["excluded"], self.report_near_dist_m)
        source = "registered_camera"
        if result is None:
            source = "map_standoff"
            result = choose_view(self._frontier_grid, pose, point, reference, scale,
                                 hist["excluded"], min(.70, self.report_near_dist_m))
        if result is None:
            hist["end"] = step
            self._note("confirm_view_unavailable", node.iid, frame=fid, reason="no_reachable_clear_view")
            return False
        waypoint, commands = result
        hist["attempts"] += 1
        self._confirm_view_state = dict(node_id=node.iid, waypoint=tuple(map(float, waypoint)), fid=fid,
                                        target=tuple(map(float, node.point[:2])), start=step,
                                        sequence=commands, cursor=0, last_step=step-1, expected=None)
        self._note("confirm_view_start", node.iid, frame=fid, waypoint=list(map(float, waypoint)),
                   target=list(map(float, node.point[:2])), planned_actions=list(commands),
                   reference_available=reference is not None, view_source=source,
                   reference_pose=None if reference is None else list(map(float,reference)),
                   attempt=hist["attempts"])
        return True

    def _confirm_view_step(self, node, obs):
        step = int(obs.step_count)
        if self._view_last_reply and self._view_last_reply[:2] == (step, node.iid):
            return self._view_last_reply[2]
        state = self._confirm_view_state
        if state and state["node_id"] != node.iid:
            return None
        if self._pending_report or self._known_owner(node) is not None:
            self._end_confirm_view("report_pending_or_registered", step)
            return None
        if getattr(self, "_goal_images", None) and self._instance_goal_index.get(node.iid) is None:
            self._end_confirm_view("goal_unbound", step)
            return None
        pose, scale = self._estimated_current_pose(), self._metric_scale_value()
        if (pose is None or scale is None or not math.isfinite(scale) or scale<=0 or
                not all(map(math.isfinite, pose)) or self._frontier_grid is None):
            self._end_confirm_view("map_or_pose_lost", step)
            return None
        pose = tuple(map(float, pose))
        if not state:
            if not self._begin_confirm_view(node, obs, pose, scale):
                return None
            state = self._confirm_view_state
        if state["cursor"]:
            last = state["sequence"][state["cursor"]-1]
            if step != state["last_step"]+1 or obs.previous_action != last:
                self._end_confirm_view("command_interrupted", step)
                return None
            if last == FORWARD and getattr(self, "_last_motion_failed", False):
                self._end_confirm_view("forward_stalled", step)
                return None
            expected = state["expected"]
            if (math.dist(expected[:2],pose[:2])*scale > .22 or
                    abs(math.remainder(pose[2]-expected[2],2*math.pi)) > math.radians(19) or
                    math.dist(node.point[:2],state["target"])*scale > .25):
                self._end_confirm_view("pose_or_target_changed", step)
                return None
        if state["cursor"] == len(state["sequence"]):
            # Use a fresh camera frame with the original same-instance and rival checks.
            ok, reason = self._report_gate(node, obs)
            self._note("confirm_view_recheck", node.iid, frame=state["fid"], live_step=step,
                       accepted=bool(ok), reason=reason, pose=list(pose))
            self._end_confirm_view("live_evidence_verified" if ok else reason, step)
            if ok:
                response = (self._stage_report(node, obs, reason), "confirm_view_report")
                self._view_last_reply = step, node.iid, response
                return response
            return None
        action = state["sequence"][state["cursor"]]
        nxt = pose_after(pose, action, scale)
        if action == FORWARD and not clear_move(self._frontier_grid, pose, nxt):
            self._end_confirm_view("route_blocked_on_updated_map", step)
            return None
        state["last_step"], state["expected"] = step, nxt
        state["cursor"] += 1
        self._note("confirm_view_action", node.iid, frame=state["fid"], action=action,
                   index=state["cursor"], total=len(state["sequence"]), pose=list(pose))
        response = action, "confirm_view_motion"
        self._view_last_reply = step, node.iid, response
        return response

    def _recover_report(self, node, obs):
        result = self._confirm_view_step(node, obs)
        return result if result is not None else super()._recover_report(node, obs)

    def _record_and_update(self, obs, action):
        result = super()._record_and_update(obs, action)
        reply = self._view_last_reply
        carrier = self._explore_follower
        # Native nav followers are advanced in act(). An ordinary explore
        # follower is advanced inside _explore_follow(), which our primitive
        # bypasses. Sync only the action that survived policy arbitration.
        if (reply and reply[0] == obs.step_count and reply[2] == (action, "confirm_view_motion")
                and carrier is not None and carrier.anchor_frame >= 0
                and (self.follower is None or self.follower.anchor_frame < 0)
                and not self._adjusting):
            carrier.dead_reckon(action)
        return result

    def _auto_report(self, obs):
        if self._confirm_view_state:
            node = self.instance_store.get(self._confirm_view_state["node_id"])
            if node is None:
                self._end_confirm_view("instance_removed", int(obs.step_count))
            else:
                reply = self._confirm_view_step(node, obs)
                if reply is not None:
                    return reply[0]
        action = super()._auto_report(obs)
        if action is not None or self._pending_report or self._auto_view_last == obs.step_count:
            return action
        self._auto_view_last = obs.step_count
        nearby = []
        for node in self.instance_store.available():
            if self._references(node) and self._known_owner(node) is None:
                distance = self._instance_distance_m(node)
                if distance is not None and distance <= 1.80:
                    nearby.append((distance, node))
        for _, node in sorted(nearby, key=lambda item: item[0]):
            reply = self._confirm_view_step(node, obs)
            if reply is not None:
                return reply[0]
        return None
