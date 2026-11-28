"""Reachable viewing poses for visual patches on an online, partially observed map.

The host supplies a map-unit (x, y, yaw) pose. All saved site and anchor coordinates
are in meters in the same oriented map frame. No instance annotations or scene mesh
are inputs. A site only describes a place to look; it is not an object detection.
"""
import heapq
import itertools
import math

import numpy as np


TURN = math.pi / 6
STEP = .25
REACH = .18
YAW_TOL = math.radians(15.01)


def _valid(grid, pose, scale):
    return (grid is not None and scale is not None and math.isfinite(scale) and scale > 0
            and pose is not None and len(pose) == 3 and all(math.isfinite(x) for x in pose))


class _Map:
    def __init__(self, grid, scale):
        self.grid = grid
        self.scale = scale
        self.cell_m = grid.res * scale
        self.cache = {}

    def cell(self, xy):
        return tuple(self.grid.world_to_cell((xy[0] / self.scale, xy[1] / self.scale)))

    def observed(self, cell):
        x, y = cell
        return self.grid.in_bounds(cell) and bool(self.grid.geometry_observed[y, x])

    def free(self, xy):
        cell = self.cell(xy)
        if cell not in self.cache:
            self.cache[cell] = self.observed(cell) and bool(self.grid.traversable(cell))
        return self.cache[cell]

    def ray_hit(self, xy, yaw, limit):
        delta = min(.07, self.cell_m * .45)
        for j in range(1, int(limit / delta) + 1):
            d = j * delta
            pt = (xy[0] + d * math.cos(yaw), xy[1] + d * math.sin(yaw))
            c = self.cell(pt)
            if not self.observed(c):
                break
            if self.grid.obstacle[c[1], c[0]]:
                return pt
        return None

    def clear_sightline(self, origin, anchor):
        # The anchor may be the first occupied voxel of a wall or table.
        distance = math.dist(origin, anchor)
        end_margin = max(.18, 1.1 * self.cell_m)
        steps = max(1, math.ceil(distance / min(.10, .55 * self.cell_m)))
        for i in range(steps + 1):
            t = i / steps
            if (1 - t) * distance <= end_margin:
                break
            xy = (origin[0] + t * (anchor[0] - origin[0]), origin[1] + t * (anchor[1] - origin[1]))
            c = self.cell(xy)
            if not self.observed(c) or self.grid.obstacle[c[1], c[0]]:
                return False
        return True

    def forward(self, pose):
        x, y, yaw = pose
        dx, dy = STEP * math.cos(yaw), STEP * math.sin(yaw)
        n = max(1, math.ceil(STEP / (.48 * self.cell_m)))
        for j in range(1, n + 1):
            if not self.free((x + dx * j/n, y + dy * j/n)):
                return None
        return x + dx, y + dy, yaw


def _pixel(patch):
    if 'uv' in patch:
        return patch['uv']
    if 'u' in patch and 'v' in patch:
        return patch['u'], patch['v']
    x0, y0, x1, y1 = patch['bbox_norm']
    return (x0+x1)/2, (y0+y1)/2


def _depth_hit(depth, u, v, focal, pitch, height, limit):
    """Choose an actual depth sample in a small central window, preserving its bearing."""
    h, w = depth.shape
    cx, cy = u*w-.5, v*h-.5
    r = max(1, min(4, round(min(w, h)*.012)))
    ys, xs = np.mgrid[max(0, round(cy)-r):min(h, round(cy)+r+1),
                      max(0, round(cx)-r):min(w, round(cx)+r+1)]
    z = depth[ys, xs]
    p = (h/2.-ys-.5) / focal
    lateral = (xs+.5-w/2.) * z / focal
    along = z * (math.cos(pitch) - p * math.sin(pitch))
    elevation = height + z * (math.sin(pitch) + p * math.cos(pitch))
    valid = (np.isfinite(z) & (z > .15) & (along > .15) &
             (np.hypot(along, lateral) <= limit) & (elevation >= .13) & (elevation <= 2.9))
    if np.count_nonzero(valid) < max(2, math.ceil(z.size*.35)):
        return None
    z, along, lateral, elevation = (a[valid] for a in (z, along, lateral, elevation))
    # Preserve foreground/background discontinuities rather than averaging them into space.
    median = float(np.median(z))
    near = np.abs(z - median)
    radius = np.hypot(xs[valid]-cx, ys[valid]-cy)
    idx = int(np.lexsort((radius, near))[0])
    return float(along[idx]), float(lateral[idx]), float(elevation[idx])


def project_patches(grid, pose, scale, patches, depth=None, hfov=90., pitch=0.,
                    camera_height=1.31, max_depth=5.):
    """Project the center camera's patch proposals. HFOV is degrees, pitch radians positive up.

    A valid RGB-D input must be aligned to the supplied RGB and contain optical-axis
    depth in meters. If no depth sensor is supplied, the conservative fallback returns
    a **bearing-column proxy**, never a measured 3-D patch point. An invalid measured
    depth is not silently replaced by that proxy.
    """
    if not _valid(grid, pose, scale):
        return []
    if not 0 < hfov < 180:
        raise ValueError('expected camera horizontal field of view in degrees')
    if depth is not None:
        depth = np.asarray(depth)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        if depth.ndim != 2 or min(depth.shape) < 3:
            raise ValueError('expected aligned HxW metric depth')
    online = _Map(grid, scale)
    x, y, yaw = pose[0]*scale, pose[1]*scale, pose[2]
    tangent = math.tan(math.radians(hfov)/2)
    result = []
    for patch in patches:
        u, v = _pixel(patch)
        if not (0 <= u < 1 and 0 <= v < 1):
            continue
        if depth is not None:
            hit = _depth_hit(depth, u, v, depth.shape[1]/(2*tangent), pitch, camera_height, max_depth)
            if hit is None:
                continue
            along, lateral, height = hit
            anchor = (x + along*math.cos(yaw) + lateral*math.sin(yaw),
                      y + along*math.sin(yaw) - lateral*math.cos(yaw))
            source = 'rgbd_measured'
        else:
            # A 2-D obstacle ray contains no vertical structure. Only near the optical
            # horizon is it useful as an approximate direction for a prospective view.
            if abs((v-.5)*2) > .42 or abs(pitch) > math.radians(18):
                continue
            ray = yaw - math.atan((2*u-1)*tangent)
            anchor = online.ray_hit((x, y), ray, max_depth)
            if anchor is None:
                continue
            height, source = None, 'observed_map_column_proxy'
        result.append(dict(patch, anchor_xy=tuple(map(float, anchor)), anchor_height_m=height,
                           anchor_source=source, source_xy=(x, y)))
    return result


def _turns(yaw, target):
    return round(math.remainder(target-yaw, math.tau)/TURN)


def _turn_plan(steps):
    return [2 if steps > 0 else 3]*abs(steps)


def _route(online, pose, target, yaw, max_actions, allow_detour):
    """Find a full witnessed primitive sequence, not a snapped or partial goal."""
    def finish(cur, route):
        turns = _turn_plan(_turns(cur[2], yaw))
        return route+turns if len(route)+len(turns) <= max_actions else None

    cur, route = pose, []
    while len(route) <= max_actions:
        if math.dist(cur[:2], target) <= REACH:
            return finish(cur, route)
        a = math.atan2(target[1]-cur[1], target[0]-cur[0])
        steps = _turns(cur[2], a)
        turn = _turn_plan(steps)
        if len(route)+len(turn)+1 > max_actions:
            break
        dest = online.forward((cur[0], cur[1], cur[2]+steps*TURN))
        if dest is None or math.dist(dest[:2], target) >= math.dist(cur[:2], target):
            break
        route.extend(turn+[1])
        cur = dest
    if not allow_detour:
        return None
    # Search in the native motor primitives. Every forward edge is checked against
    # the current observed occupancy; no navmesh or hidden free cell is consulted.
    counter = itertools.count()
    def key(p):
        return round(p[0]/.085), round(p[1]/.085), round((p[2]-pose[2])/TURN) % 12
    def estimate(p):
        return max(0., math.dist(p[:2], target)-REACH)/STEP
    queue = [(estimate(pose), next(counter), pose, ())]
    best = {key(pose):0}
    for _ in range(1050):
        if not queue:
            break
        _, _, cur, actions = heapq.heappop(queue)
        n = len(actions)
        if best[key(cur)] < n:
            continue
        if math.dist(cur[:2], target) <= REACH:
            final = finish(cur, list(actions))
            if final is not None:
                return final
        if n+1 > max_actions:
            continue
        for act in (1, 2, 3):
            if act == 1:
                dest = online.forward(cur)
                if dest is None:
                    continue
            else:
                dest = cur[0], cur[1], cur[2]+(TURN if act == 2 else -TURN)
            remaining = estimate(dest)
            state = key(dest)
            if n+1+remaining > max_actions or best.get(state, math.inf) <= n+1:
                continue
            best[state] = n+1
            heapq.heappush(queue, (n+1+1.25*remaining, next(counter), dest, actions+(act,)))
    return None


def plan_to_site(grid, pose, scale, site, budget=32, max_leg_steps=6):
    """Replan to the fixed metric site from the latest pose, returning a short executable leg."""
    if not _valid(grid, pose, scale):
        return dict(status='blocked', reason='map_unavailable', actions=(), distance_m=None)
    online = _Map(grid, scale)
    cur = float(pose[0]*scale), float(pose[1]*scale), float(pose[2])
    target, yaw = site['site_xy'], site['site_yaw']
    distance = math.dist(cur[:2], target)
    def decision(status, reason, actions=()):
        return dict(status=status, reason=reason, actions=tuple(actions), distance_m=distance)
    if not online.free(target):
        return decision('blocked', 'site_untraversable')
    anchor = site['anchor_xy']
    if not online.clear_sightline(target, anchor):
        return decision('blocked', 'sightline_lost')
    if distance <= REACH and abs(math.remainder(yaw-cur[2], math.tau)) <= YAW_TOL:
        if not online.clear_sightline(cur[:2], anchor):
            return decision('blocked', 'sightline_lost_at_current_pose')
        return decision('arrived', 'position_and_heading')
    if budget < 1 or max_leg_steps < 1:
        return decision('budget', 'no_action_budget')
    route = _route(online, cur, target, yaw, min(42, int(budget)), True)
    if route is None:
        return decision('blocked', 'no_certified_route_within_budget')
    return decision('active', 'approach_site', route[:int(max_leg_steps)])


def propose_sites(grid, pose, scale, anchors, budget=32, max_sites=12):
    """Produce diverse, fully reachable stand-off poses in the order of the supplied cues.

    'actions' is the full certified route when the site was proposed; plan_to_site
    replans a short leg when the site is chosen, including after the map updates.
    """
    if not _valid(grid, pose, scale) or budget < 1 or max_sites < 1:
        return []
    online = _Map(grid, scale)
    cur = float(pose[0]*scale), float(pose[1]*scale), float(pose[2])
    unique, selected = [], set()
    for patch in anchors:
        anchor = patch['anchor_xy']
        if any(math.dist(anchor, old['anchor_xy']) < .34 for old in unique):
            continue
        unique.append(patch)
    rows = []
    for radius, offset in ((.95, 0.), (.95, 32.), (.95, -32.), (1.4, 0.),
                           (.72, 0.), (1.20, 55.), (1.20, -55.)):
        for patch in unique:
            anchor = patch['anchor_xy']
            towards = math.atan2(cur[1]-anchor[1], cur[0]-anchor[0])+math.radians(offset)
            target = (anchor[0]+radius*math.cos(towards), anchor[1]+radius*math.sin(towards))
            if math.dist(cur[:2], target) < .34 or not online.free(target):
                continue
            yaw = math.atan2(anchor[1]-target[1], anchor[0]-target[0])
            key = (round(target[0]/.22), round(target[1]/.22), round(yaw/TURN) % 12)
            if key in selected or not online.clear_sightline(target, anchor):
                continue
            # Cheap direct routes are used to generate candidates. If an obstacle
            # subsequently appears, the persistent replanner can also go around it.
            route = _route(online, cur, target, yaw, min(32,int(budget)), False)
            if route is None or not route or 1 not in route:
                continue
            selected.add(key)
            surface_key = (round(anchor[0]/.4), round(anchor[1]/.4))
            rows.append(dict(patch, key='local/%s/%s/%s'%key, source='local_observation_site',
                             position=target, heading=yaw, site_xy=target, site_yaw=yaw,
                             actions=tuple(route), steps=len(route), visible_cells={surface_key},
                             novelty=0., stand_off_m=radius))
            if len(rows) >= max_sites:
                return rows
    return rows
