"""Find short executable legs toward new regions beyond the current camera cone."""
from collections import deque
import math

import numpy as np

from map_views import endpoint, _step


def frontier_plans(grid, pose, scale, budget):
    """BFS only over online free cells; distant map boundaries compete across bearings."""
    horizon = min(24, int(budget))
    if horizon < 4 or grid is None or not scale or scale <= 0:
        return []
    start = tuple(grid.world_to_cell(pose[:2]))
    delta = grid.res * scale
    seen = grid.geometry_observed
    h, w = seen.shape
    radius = math.ceil(10 / delta)
    x0, y0 = max(0, start[0]-radius), max(0, start[1]-radius)
    x1, y1 = min(w, start[0]+radius+1), min(h, start[1]+radius+1)
    if not x0 <= start[0] < x1 or not y0 <= start[1] < y1:
        return []
    # Integral image counts genuinely unobserved cells in a metric neighborhood.
    unknown = (~seen[y0:y1, x0:x1]).astype(np.int32)
    counts = np.pad(unknown, ((1,0),(1,0))).cumsum(0).cumsum(1)
    margin = max(2, round(.75 / delta))
    def unseen(x, y):
        a, b = x-x0, y-y0
        lo, left = max(0,b-margin), max(0,a-margin)
        hi, right = min(y1-y0,b+margin+1), min(x1-x0,a+margin+1)
        return (counts[hi,right]-counts[lo,right]-counts[hi,left]+counts[lo,left])/((hi-lo)*(right-left))
    def world(cell):
        if hasattr(grid, "cell_to_world"):
            return grid.cell_to_world(cell)
        return grid.origin + (np.array(cell) + .5) * grid.res

    parent = {start:None}
    queue = deque([(start,0)])
    per_bearing = {}
    tested = {start}
    while queue:
        node, n = queue.popleft()
        if n * delta >= 12.:
            continue
        x,y = node
        free_neighbors = 0
        for dx,dy in ((1,0),(0,1),(-1,0),(0,-1)):
            nx,ny=x+dx,y+dy
            if not (x0 <= nx < x1 and y0 <= ny < y1):
                continue
            free_neighbors += int(not seen[ny,nx])
            child=(nx,ny)
            if child not in tested:
                tested.add(child)
                if grid.traversable(child):
                    parent[child]=node
                    queue.append((child,n+1))
        distance = math.hypot(x-start[0],y-start[1]) * delta
        if free_neighbors == 0 or distance < 1.25:
            continue
        gain = unseen(x,y)
        if gain < .18:
            continue
        bearing = math.atan2(y-start[1],x-start[0])
        sector = round((bearing-pose[2])/(math.pi/4)) % 8
        utility = gain * min(distance, 6.) / (1. + n*delta/12.)
        if sector not in per_bearing or utility > per_bearing[sector][0]:
            per_bearing[sector] = utility, node

    result = []
    for sector, (_, goal) in sorted(per_bearing.items()):
        cells = []
        node = goal
        while node is not None:
            cells.append(node)
            node = parent[node]
        path = [world(c) for c in reversed(cells)]
        actions, current, cursor = [], tuple(pose), 0
        while len(actions) < horizon and cursor < len(path):
            while cursor < len(path) and math.dist(current[:2], path[cursor]) * scale <= .32:
                cursor += 1
            if cursor == len(path):
                break
            tx,ty = path[cursor]
            goal_heading = math.atan2(ty-current[1], tx-current[0])
            turn = round(math.remainder(goal_heading-current[2], math.tau)/(math.pi/6))
            turn_plan = [2 if turn>0 else 3] * abs(turn)
            if len(actions) + len(turn_plan) + 1 > horizon:
                break
            movement = turn_plan + [1]
            dest = endpoint(grid,current,scale,movement)
            if dest is None:
                break
            actions.extend(movement)
            current = dest
        if actions.count(1) >= 3:
            result.append((f"global-{sector}",actions))
    return result
