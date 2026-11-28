"""Short-lived geometry from measured blocked forward motions in the SLAM frame."""
import math


class MotionMemory:
    def __init__(self):
        self.contacts = {}

    def record(self, pose, scale, tick):
        x, y, yaw = pose
        # Leave the actual camera cell free so the native follower can turn and retreat.
        cx, cy = x * scale + .23 * math.cos(yaw), y * scale + .23 * math.sin(yaw)
        self.contacts[round(cx / .12), round(cy / .12)] = (cx, cy, tick)

    def grid(self, grid, scale, tick):
        self.contacts = {k: v for k, v in self.contacts.items() if tick - v[2] < 48}
        if not self.contacts:
            return grid
        blocked = set()
        for x,y,_ in self.contacts.values():
            cell = grid.world_to_cell((x/scale, y/scale))
            blocked.add(tuple(cell))
        return ContactGrid(grid, blocked)


class ContactGrid:
    def __init__(self, original, blocked):
        self.original, self.blocked = original, blocked
    def traversable(self, cell):
        return tuple(cell) not in self.blocked and self.original.traversable(cell)
    def __getattr__(self, key):
        return getattr(self.original, key)
