"""Carry an accepted exploration view through the native exploration follower."""
import math
from numbers import Integral
from copy import copy

import numpy as np

from agents.navigator import PathFollower


def route_geometry(pose, scale, actions):
    """Construct the selected view and its ordered approach waypoints in the SLAM frame."""
    if not actions or not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid scale or empty approach")
    x, y, yaw = map(float, pose)
    if not all(map(math.isfinite, (x, y, yaw))):
        raise ValueError("invalid source pose")
    path = [(x, y)]
    for action in actions:
        if isinstance(action, bool) or not isinstance(action, Integral) or action not in (1, 2, 3):
            raise ValueError("expected a native navigation primitive")
        if action == 1:
            x += math.cos(yaw) * .25 / scale
            y += math.sin(yaw) * .25 / scale
            path.append((x, y))
        else:
            yaw += math.pi / 6 if action == 2 else -math.pi / 6
    return path, yaw


class ViewFollower(PathFollower):
    """Use the native waypoint controller; require the camera to face the requested view."""
    def __init__(self, pose, scale, actions, anchor=-1):
        super().__init__(scale=scale, reach_m=.125 + 1.e-7, waypoint_m=.125 + 1.e-7)
        path, self.final_yaw = route_geometry(pose, scale, actions)
        self.set_path(path)
        self.x, self.y, self.yaw = map(float, pose)
        self.anchor_frame = anchor
        self.outcome = None
        self.goal_xy = path[-1]
        self.repairs = 0
        self.staging = False

    def detour(self, grid, pose, blocked_points=()):
        """Replan on a private copy of the observed grid. Never snap to a different goal."""
        if grid is None or self.repairs >= 2:
            return False
        local = copy(grid)
        local.obstacle = grid.obstacle.copy()
        radius = .16 / self.scale
        for point in blocked_points:
            col, row = grid.world_to_cell(point)
            cells = int(math.ceil(radius/grid.res))
            for y in range(max(0,row-cells), min(local.obstacle.shape[0],row+cells+1)):
                for x in range(max(0,col-cells), min(local.obstacle.shape[1],col+cells+1)):
                    if math.dist(local.cell_to_world((x,y)), point) <= radius:
                        local.obstacle[y,x] = True
        path = local.astar(pose[:2], self.goal_xy, snap_radius=2)
        self.staging = not path or math.dist(path[-1], self.goal_xy)*self.scale > .125
        if self.staging:
            # If the intended view is beyond the current map, first move to a reachable
            # observation point on the known side. A stage is never an arrival receipt.
            ys,xs = np.nonzero(local.free & ~local.obstacle)
            xy = local.origin + (np.column_stack((xs,ys))+.5)*local.res
            distance = np.linalg.norm(xy-np.asarray(self.goal_xy),axis=1)*self.scale
            allowed = np.nonzero((distance < math.dist(pose[:2],self.goal_xy)*self.scale-.35)
                                 & (np.linalg.norm(xy-np.asarray(pose[:2]),axis=1)*self.scale >= .35))[0]
            buckets=set()
            path=None
            for i in allowed[np.argsort(distance[allowed])]:
                bucket=tuple(np.floor(xy[i]*self.scale/.3).astype(int))
                if bucket in buckets:
                    continue
                buckets.add(bucket)
                if len(buckets)>32:
                    break
                path=local.astar(pose[:2],xy[i],snap_radius=1)
                if path:
                    break
            if not path:
                return False
        path = list(local.shortcut(path))
        if not self.staging and math.dist(path[-1], self.goal_xy) > 1.e-6:
            path.append(self.goal_xy)
        self.x, self.y, self.yaw = map(float, pose)
        self.set_path(path)
        self.repairs += 1
        self.outcome = None
        return True

    def next_action(self):
        action, arrived = super().next_action()
        if arrived:
            if self.staging:
                self.outcome = "intermediate_view_only"
                return None, True
            angle = math.remainder(self.final_yaw - self.yaw, math.tau)
            if abs(angle) > math.pi / 12 + 1.e-8:
                return (2 if angle > 0 else 3), False
            self.outcome = "estimated_view_reached"
        elif action is None:
            self.outcome = "native_path_lost"
        return action, arrived


class FocusViewMixin:
    """Mixin before a MOST-Nav host. Normal navigation, cancellation and report take priority."""
    _focus_view = None

    def _focus_record(self, event, **fields):
        if hasattr(self, "emit"):
            self.emit(event, **fields)
        elif hasattr(self, "_log_event"):
            self._log_event(f"{event} {fields}")

    def start_focus_view(self, observation, actions, tag):
        existing = self._explore_follower
        if existing is None or getattr(self, "mode", None) != "explore":
            return False
        pose = self._estimated_current_pose()
        if pose is None:
            pose = getattr(self, "_frontier_pose", None)
        if pose is None:
            return False
        remaining = observation.max_steps - observation.step_count
        if remaining < len(actions) or not actions:
            return False
        scale = float(existing.scale)
        follower = ViewFollower(pose, scale, actions, existing.anchor_frame)
        # Snapshot the old hook only through its owner's public cancellation method.
        # The old carrier must not survive alongside the view's carrier.
        if getattr(self, "commit", None) is not None:
            self._close_commit("selected_view_activated")
        self._finish_focus("superseded")
        self._explore_follower = follower
        self._focus_view = (follower, tag, observation.step_count,
                            min(observation.max_steps, observation.step_count + len(actions) + 8))
        self._focus_record("focus_view_start", tag=tag, requested_actions=len(actions))
        return True

    def _finish_focus(self, reason):
        if self._focus_view is not None:
            follower, tag, start, _ = self._focus_view
            self._focus_view = None
            self._focus_record("focus_view_end", tag=tag, start=start, reason=reason)

    def _explore_follow(self, observation):
        active = self._focus_view
        if active is not None:
            follower, _, _, deadline = active
            if self._explore_follower is not follower:
                self._finish_focus("native_recovery_or_replacement")
            elif observation.step_count >= deadline:
                follower.set_path(None)
                self._finish_focus("view_budget_exhausted")
            elif follower.staging and follower.distance_to_goal()*follower.scale <= .25:
                self._plan_exploration(observation,select=False)
                pose=getattr(self,"_frontier_pose",None)
                if pose is None or not follower.detour(self._frontier_grid, pose,
                       [p for p,end in self._nav_blocked_points if end>=observation.step_count]):
                    follower.set_path(None)
                    self._finish_focus("intermediate_view_only")
        action = super()._explore_follow(observation)
        if self._focus_view is not None and self._focus_view[0].outcome is not None:
            follower = self._focus_view[0]
            reason = follower.outcome
            if reason == "estimated_view_reached":
                frame = (getattr(self, "_last_feed_info", {}) or {}).get("frame_id")
                if frame is not None and follower.anchor_frame == frame:
                    reason = "current_map_pose_view_reached"
            self._finish_focus(reason)
        return action

    def _explore_action(self, observation):
        active = self._focus_view
        if (not active or not self._last_motion_failed or
                self._explore_follower is not active[0] or observation.step_count >= active[3]):
            return super()._explore_action(observation)
        follower = active[0]
        # Only a revised map and a recent source keyframe may justify continuing after a stall.
        revision = getattr(self, "_frontier_revision", -1)
        self._plan_exploration(observation, select=False)
        pose = getattr(self, "_frontier_pose", None)
        current_fid = (getattr(self, "_last_feed_info", {}) or {}).get("frame_id")
        poses, ids = self.client.get_all_poses()
        last_fid = max(ids) if poses is not None and ids is not None and len(ids) else None
        fresh = (pose is not None and self._frontier_revision > revision and last_fid is not None
                 and current_fid is not None and 0 <= current_fid-last_fid <= 2)
        if fresh:
            self._block_failed_nav_direction(observation, follower)
            blocks = [point for point, end in self._nav_blocked_points if end >= observation.step_count]
            if follower.detour(self._frontier_grid, pose, blocks):
                follower.anchor_frame = last_fid
                self._focus_record("focus_view_detour", count=follower.repairs)
                # The native collision is already consumed. Preserve native goal interrupts and
                # ordinary exploration dispatch for this very action.
                self._last_motion_failed = False
                try:
                    return super()._explore_action(observation)
                finally:
                    self._last_motion_failed = True
        self._finish_focus("no_reachable_detour_with_fresh_map")
        return super()._explore_action(observation)

    def act(self, observation):
        action = super().act(observation)
        if self._focus_view and self._explore_follower is not self._focus_view[0]:
            self._finish_focus("native_recovery_or_replacement")
        return action

    def init_episode(self, observation):
        self._finish_focus("episode_reset")
        return super().init_episode(observation)
