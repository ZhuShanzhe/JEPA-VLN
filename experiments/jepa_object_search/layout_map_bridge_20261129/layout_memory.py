"""Observation-only metric layout, scoped to one episode, floor and inspection origin.

x points right in the first camera, z forward, yaw positive to the left (radians).
Cells describe body clearance: 0 unknown, 1 observed free, 2 blocked.
"""
import numpy as np


CELL = 0.125
RADIUS = 64
IDS = np.arange(-RADIUS, RADIUS + 1)
DX, DZ = np.meshgrid(IDS * CELL, IDS * CELL)


class OccupancyPatch:
    cell_m = CELL

    def __init__(self, x, z, episode, floor, frame_id):
        self.scope = (str(episode), str(floor))
        self.frame_id = frame_id
        self.cx, self.cz = np.rint(np.asarray([x, z]) / CELL) * CELL
        self.cells = np.zeros(DX.shape, np.uint8)

    def query(self, x, z):
        x, z = np.broadcast_arrays(x, z)
        valid = np.isfinite(x) & np.isfinite(z)
        i = np.rint((np.where(valid, x, self.cx) - self.cx) / CELL).astype(np.int64) + RADIUS
        j = np.rint((np.where(valid, z, self.cz) - self.cz) / CELL).astype(np.int64) + RADIUS
        inside = valid & (i >= 0) & (j >= 0) & (i <= 2*RADIUS) & (j <= 2*RADIUS)
        return np.where(inside, self.cells[j.clip(0, 2*RADIUS), i.clip(0, 2*RADIUS)], 0)


class LayoutMap:
    @staticmethod
    def from_depth_history(history, *, episode, floor=0, frame_id=None, external=None):
        """Each history is one contiguous local inspection, including all real contact events.

        external is a matching, fresh MOST patch in the same inspection coordinate frame.
        Its free/occupied evidence is filled first, then updated with current RGB-D geometry.
        """
        last = history[-1] if history else {"x": 0., "z": 0.}
        patch = OccupancyPatch(last["x"], last["z"], episode, floor, frame_id)
        xx, zz = DX + patch.cx, DZ + patch.cz
        if external is not None and external.scope == patch.scope and frame_id is not None and external.frame_id == frame_id:
            patch.cells[:] = external.query(xx, zz)
        local = np.zeros_like(patch.cells)
        bumps = np.zeros_like(patch.cells, dtype=bool)
        previous = None
        for frame in history:
            if (str(frame.get("episode", episode)), str(frame.get("floor", floor))) != patch.scope:
                raise ValueError("mixed episode or floor in observation history")
            x, z, yaw = float(frame["x"]), float(frame["z"]), float(frame["yaw"])
            if not np.isfinite([x, z, yaw]).all():
                raise ValueError("invalid odometry")
            if frame.get("bump", False) and previous is not None:
                px, pz, heading = previous
                bx, bz = px - .25*np.sin(heading), pz + .25*np.cos(heading)
                bumps |= (xx - bx)**2 + (zz - bz)**2 < .14**2
            previous = (x, z, yaw)
            depth = frame.get("depth")
            if depth is None:
                continue
            depth = np.asarray(depth).squeeze()
            if depth.shape != (192, 192):
                raise ValueError("requires aligned metric 192-square, 90-degree planar depth")
            # Eight oblique rays per azimuth sample the robot's vertical clearance.
            rows = np.arange(77, 178, 8)
            samples = depth[rows].astype(np.float64)
            valid = np.isfinite(samples) & (samples > .06) & (samples < 15.)
            height = 1.31 - (rows[:, None] + .5 - 96.) / 96. * samples
            solid = valid & (height >= .12) & (height <= 1.65)
            obstacle = np.min(np.where(solid, samples, np.inf), axis=0)
            # Limit clearing to within the body-height band or rays that reach the floor.
            usable = valid & (height >= -.12) & (height <= 1.65)
            extent = np.minimum(obstacle, np.max(np.where(usable, samples, 0.), axis=0))
            s, c = np.sin(yaw), np.cos(yaw)
            mx, mz = xx-x, zz-z
            side, front = mx*c + mz*s, -mx*s + mz*c
            u = np.floor(96. + 96.*side / np.maximum(front, .01)).astype(int)
            seen = (front > .06) & (u >= 0) & (u < 192)
            u = u.clip(0, 191)
            local[seen & (front < extent[u] - .065)] = 1
            local[seen & (np.abs(front - obstacle[u]) <= .08)] = 2
        # Inflate measured surfaces once for footprint clearance; native MOST is already inflated.
        blocked = np.pad(local == 2, 1)
        h, w = local.shape
        inflated = np.logical_or.reduce([blocked[a:a+h,b:b+w] for a in range(3) for b in range(3)])
        patch.cells[(local == 1) & (patch.cells != 2)] = 1
        patch.cells[inflated | bumps] = 2
        return patch
