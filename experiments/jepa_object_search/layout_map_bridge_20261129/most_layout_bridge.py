"""Read the live MOST navigation agent's atomic frontier snapshot for local JEPA queries."""
import numpy as np
from layout_memory import LayoutMap, OccupancyPatch, DX, DZ


def layout_for_history(history):
    current = history[-1] if history else {}
    return LayoutMap.from_depth_history(history, episode=current.get("episode", "local"),
                                        floor=current.get("floor", 0), frame_id=current.get("frame_id"),
                                        external=current.get("layout_patch"))


class MostLayoutBridge:
    """Single-floor inspection bridge; native maps, poses and revisions are never persisted."""

    def __init__(self):
        self.scope = None
        self.last_step = None
        self.last_revision = None
        self.last_status = "empty"

    def snapshot(self, agent, observation, local_pose, *, floor=0, refresh=True):
        episode = str(observation.episode_id)
        scope = (episode, str(floor))
        step = int(observation.step_count)
        valid_pose = local_pose is not None and np.shape(local_pose) == (3,) and np.isfinite(local_pose).all()
        center = local_pose[:2] if valid_pose else (0., 0.)
        patch = OccupancyPatch(*center, episode, floor, step)
        def reject(reason):
            self.last_status = reason
            return patch

        if str(agent.episode_id) != episode:
            return reject("episode_mismatch")
        if not getattr(agent, "pose_free", False):
            return reject("requires_pose_free_metric_mapping")
        if not valid_pose:
            return reject("missing_odometry")
        # Multi-floor correspondence cannot be determined from a projected 2D frontier.
        if self.scope and self.scope[0] == episode and self.scope[1] != str(floor):
            return reject("needs_single_floor_submap")
        if self.scope != scope:
            self.last_step = self.last_revision = None
            self.scope = scope
        if self.last_step is not None and step < self.last_step:
            return reject("out_of_order_observation")
        if refresh and getattr(agent, "_last_frontier_step", None) != step:
            agent._ensure_decision_map_snapshot(observation, force=True)
        if getattr(agent, "_last_frontier_step", None) != step:
            return reject("no_fresh_snapshot")
        grid = getattr(agent, "_frontier_grid", None)
        pose = getattr(agent, "_frontier_pose", None)
        scale = getattr(agent, "_frontier_scale", None)
        revision = getattr(agent, "_frontier_revision", None)
        if grid is None or pose is None or scale is None or revision is None:
            return reject("no_map_pose_pair")
        if self.last_revision is not None and (revision < self.last_revision or
                (step != self.last_step and revision == self.last_revision)):
            return reject("stale_revision")
        if not np.isfinite(pose).all() or not np.isfinite(scale) or scale <= 0:
            return reject("invalid_metric_pose")
        if not np.isclose(grid.unit_per_m * scale, 1., rtol=.001):
            return reject("metric_scale_mismatch")
        lx, lz, lyaw = local_pose
        gx, gy, gyaw = pose
        # Local odometry yaw=0 points forward (+z); native yaw=0 points +map X.
        ax, az = DX + patch.cx - lx, DZ + patch.cz - lz
        right = ax*np.cos(lyaw) + az*np.sin(lyaw)
        front = -ax*np.sin(lyaw) + az*np.cos(lyaw)
        wx = gx + (front*np.cos(gyaw) + right*np.sin(gyaw)) / scale
        wy = gy + (front*np.sin(gyaw) - right*np.cos(gyaw)) / scale
        ix = np.floor((wx-grid.origin[0])/grid.res).astype(int)
        iy = np.floor((wy-grid.origin[1])/grid.res).astype(int)
        inside = (ix >= 0) & (iy >= 0) & (ix < grid.free.shape[1]) & (iy < grid.free.shape[0])
        row, col = iy[inside], ix[inside]
        seen_free = grid.geometry_observed[row,col] & grid.free[row,col]
        occupied = grid.obstacle[row,col]
        patch.cells[inside] = np.where(occupied, 2, seen_free.astype(np.uint8))
        if getattr(agent, "_frontier_revision", None) != revision:
            patch.cells.fill(0)
            return reject("map_changed_during_read")
        self.last_step, self.last_revision = step, revision
        self.last_status = "observed" if patch.cells.any() else "no_observed_cells"
        return patch

    def read(self, agent, observation, history, *, floor=0):
        """No RGB-D required in the native arm. Supplied depth must match local camera calibration."""
        current = history[-1]
        local_pose = tuple(current[key] for key in ("x", "z", "yaw"))
        external = self.snapshot(agent, observation, local_pose, floor=floor)
        return LayoutMap.from_depth_history(history, episode=observation.episode_id, floor=floor,
                                            frame_id=observation.step_count, external=external)


class MostLayoutInspection:
    """Bind one upstream observation to each downstream observation before route inference."""

    def __init__(self, inspection):
        self.inspection = inspection

    def run(self, agent, observe, move, source_box, forward_steps, budget, directory, *, floor=0):
        """observe() or move(action) supplies (MOST observation, calibrated local RGB-D+odometry).

        Host must update the native mapper before returning the pair for the same step.
        A floor change requires a new single-floor submap and local inspection. No simulator
        object IDs, reference navigation mesh, or full map are read by this interface.
        """
        bridge = MostLayoutBridge()
        expected_episode = None
        last_frame = None
        def bind(pair):
            nonlocal expected_episode, last_frame
            observation, local = pair
            episode, step = str(observation.episode_id), int(observation.step_count)
            if expected_episode is not None and (episode != expected_episode or step != last_frame + 1):
                raise ValueError("local inspection interrupted or upstream observation out of order")
            if local.get("frame_id",step) != step or str(local.get("episode",episode)) != episode:
                raise ValueError("map and sensor observations must have matching provenance")
            if str(local.get("floor",floor)) != str(floor):
                raise ValueError("start a new single-floor inspection")
            expected_episode, last_frame = episode, step
            patch = bridge.snapshot(agent, observation, tuple(local[k] for k in ("x","z","yaw")),
                                    floor=floor)
            return dict(local, episode=episode, frame_id=step, floor=floor, layout_patch=patch)
        return self.inspection.run(bind(observe()), source_box, forward_steps, budget,
                                   lambda action: bind(move(action)), directory)
