"""Propose camera sites using the current navigation grid, without scene annotations."""
import math
from collections import defaultdict


def _step(pose, act, scale):
    x, y, yaw = pose
    if act == 1:
        return x + .25 * math.cos(yaw) / scale, y + .25 * math.sin(yaw) / scale, yaw
    if act not in (2, 3):
        raise ValueError("unknown navigation action")
    return x, y, yaw + (math.pi / 6 if act == 2 else -math.pi / 6)


def endpoint(grid, pose, scale, plan):
    for act in plan:
        target = _step(pose, act, scale)
        if act == 1:
            n = max(1, math.ceil(.25 / (grid.res * scale * .5)))
            for i in range(1, n + 1):
                frac = i / n
                x, y = (pose[k] + frac * (target[k] - pose[k]) for k in (0, 1))
                if not grid.traversable(grid.world_to_cell((x, y))):
                    return None
        pose = target
    return pose


def _rays(grid, pose, scale):
    """Known obstacle returns only; geometry pose quantization remains stable if the grid grows."""
    x, y, yaw = pose
    increment = max(.04, min(.15, grid.res * scale * .65))
    n = math.ceil(3.0 / increment)
    surfaces, unknown = set(), 0
    for k in range(-4, 5):
        a = yaw + k * math.pi / 18
        for j in range(1, n + 1):
            d = j * increment / scale
            mx, my = x + math.cos(a) * d, y + math.sin(a) * d
            cx, cy = grid.world_to_cell((mx, my))
            if not grid.in_bounds((cx, cy)):
                unknown += 1
                break
            if grid.obstacle[cy, cx]:
                surfaces.add((round(mx * scale / .4), round(my * scale / .4)))
                break
            if not grid.geometry_observed[cy, cx]:
                unknown += 1
                break
    return surfaces, unknown / 9.


def visible_map_surfaces(grid, pose, scale):
    return _rays(grid, pose, scale)[0]


def _turns(count):
    return [2 if count > 0 else 3] * abs(count)


def candidates(grid, pose, scale, native_plans, budget):
    """Reachable prefixes of long frontiers plus short translated views into side areas."""
    if not grid or not pose or not scale or not math.isfinite(scale) or scale <= 0:
        return []
    horizon = min(24, int(budget))
    if horizon < 1:
        return []
    rows, occupied = [], set()
    def add(key, plan, source):
        if not plan or len(plan) > horizon or 1 not in plan:
            return
        p = endpoint(grid, pose, scale, plan)
        if p is None:
            return
        position = p[0] * scale, p[1] * scale
        # Keep separate views of different sides of the same table or wall.
        cell = (round(position[0] / .4), round(position[1] / .4), round(p[2] / (math.pi / 3)) % 6)
        if cell in occupied:
            return
        surfaces, novelty = _rays(grid, p, scale)
        rows.append(dict(key=key, source=source, position=position, heading=p[2],
                         actions=tuple(plan), steps=len(plan), visible_cells=surfaces, novelty=novelty))
        occupied.add(cell)

    for name, plan in native_plans:
        # A longer native trajectory with an unsafe end can still contain a reachable intermediate view.
        for n in sorted({min(len(plan), horizon), min(len(plan), horizon, 8)}, reverse=True):
            add(f"{name}/{n}", list(plan[:n]), "native_frontier")
    base = len(rows)
    # From known free floor, approach possible support surfaces or obtain a new line of sight.
    # Neither category examples nor test goal geometry are inputs to this proposal stage.
    for bearing in (0, -2, 2, -4, 4, 6):
        lead = _turns(bearing)
        for forward in (5, 10):
            walk = lead + [1] * forward
            for side in (0, -2, 2):
                add(f"view/{bearing}/{forward}/{side}", walk + _turns(side), "map_view")
            for side in (-3, 3):
                add(f"bend/{bearing}/{forward}/{side}", walk + _turns(side) + [1] * 4, "map_bend")
    # Limit model compute with balanced travel bearings; keep native candidates unchanged.
    groups = defaultdict(list)
    for row in rows[base:]:
        x, y = row["position"]
        bearing = math.atan2(y - pose[1] * scale, x - pose[0] * scale)
        sector = round((bearing - pose[2]) / (math.pi / 3)) % 6
        groups[sector].append(row)
    for group in groups.values():
        group.sort(key=lambda r: (-(r["novelty"] + bool(r["visible_cells"])), -r["steps"]))
    return rows[:base] + [group[i] for i in range(4) for group in groups.values() if i < len(group)]
