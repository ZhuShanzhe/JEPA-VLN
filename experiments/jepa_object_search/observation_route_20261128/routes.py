"""Reachable short viewing motions, scored only with the robot's online observations.

The host supplies a pose in grid units, a meter/unit scale, and the actual continuation
of its regional planner. XY values in returned candidates and optional cue are meters.
Unknown regions may be viewed from a safe point. They never certify a forward edge.
"""

import math

from observation_sites import _Map, TURN


def _visibility(online, pose):
    """Nine horizontal rays. Return measured surface bins and first unknown bins separately."""
    surfaces, frontiers = set(), set()
    increment = max(.04, min(.15, online.cell_m * .65))
    for offset in range(-4, 5):
        angle = pose[2] + math.radians(10 * offset)
        dx, dy = increment * math.cos(angle), increment * math.sin(angle)
        for depth in range(1, int(3.0 / increment) + 1):
            xy = pose[0] + depth * dx, pose[1] + depth * dy
            key = round(xy[0] / .4), round(xy[1] / .4)
            cell = online.cell(xy)
            if not online.grid.in_bounds(cell):
                break
            if not online.observed(cell):
                frontiers.add(key)
                break
            if online.grid.obstacle[cell[1], cell[0]]:
                surfaces.add(key)
                break
    return surfaces, frontiers


def _end(online, start, actions):
    pose = start
    for act in actions:
        if act == 1:
            pose = online.forward(pose)
            if pose is None:
                return None
        else:
            pose = pose[0], pose[1], pose[2] + (TURN if act == 2 else -TURN)
    return pose


def _turn(n):
    return (2 if n > 0 else 3,) * abs(n)


def _primitives(limit):
    # Camera in-place turns and translated poses are distinct controls.
    for angle in (-1, 1, -2, 2, -3, 3, -4, 4):
        if abs(angle) <= limit:
            yield _turn(angle)
    for first in (0, -1, 1, -2, 2, -3, 3):
        for forward in (1, 2):
            for last in (0, -1, 1, -2, 2):
                actions = _turn(first) + (1,) * forward + _turn(last)
                if len(actions) <= limit:
                    yield actions


def _centered_evidence(online, pose, anchor):
    # An observed obstruction strictly before the anchor rules out this view. An
    # unknown sightline remains an intermediate option rather than proof of visibility.
    distance = math.dist(pose[:2], anchor)
    bearing = math.atan2(anchor[1] - pose[1], anchor[0] - pose[0])
    angle = abs(math.remainder(bearing - pose[2], math.tau))
    if angle >= math.pi / 4 or distance < .30:
        return 0.
    steps = max(1, math.ceil(distance / min(.12, online.cell_m * .65)))
    unknown = False
    for i in range(1, steps):
        if distance * (1 - i / steps) < max(.18, 1.1 * online.cell_m):
            break
        pt = tuple(pose[k] + (anchor[k] - pose[k]) * i / steps for k in (0, 1))
        cell = online.cell(pt)
        if not online.observed(cell):
            unknown = True
            continue
        if online.grid.obstacle[cell[1], cell[0]]:
            return 0.
    centering = 1 - angle / (math.pi / 4)
    return centering * (.5 if unknown else 1.) / max(.75, distance)


def observation_routes(grid, pose, scale, *, mode, region_actions, budget, observed_surfaces=(),
                       anchor_xy=None, scorer=None, horizon=4):
    if mode not in ('discover', 'verify'):
        raise ValueError('expected discover or verify')
    if mode == 'verify' and (anchor_xy is None or len(anchor_xy) != 2
                             or not all(math.isfinite(v) for v in anchor_xy)):
        raise ValueError('a finite observed cue in map meters is required')
    if not 1 <= horizon <= 4:
        raise ValueError('supported viewing horizon is one to four actions')
    n = min(int(budget), horizon)
    base = tuple(region_actions[:max(0, n)])
    result = dict(mode=mode, candidates=[], continue_id='continue', selected_id='continue',
                  geometry_id='continue', status='no_safe_continuation')
    if not base or grid is None or pose is None or len(pose) != 3 or scale is None:
        return result
    if not (math.isfinite(scale) and scale > 0 and all(math.isfinite(v) for v in pose)):
        return result
    if any(a not in (1, 2, 3) for a in base):
        return result
    online = _Map(grid, scale)
    start = float(pose[0] * scale), float(pose[1] * scale), float(pose[2])
    end = _end(online, start, base)
    if end is None:
        return result
    remembered = {tuple(key) for key in observed_surfaces}
    frontier_now = set()
    for yaw in (-TURN, 0., TURN):
        surface, front = _visibility(online, (*start[:2], start[2] + yaw))
        remembered.update(surface)
        frontier_now.update(front)

    def candidate(actions, end, name):
        surfaces, front = _visibility(online, end)
        gained, boundaries = sorted(surfaces - remembered), sorted(front - frontier_now)
        if mode == 'verify':
            utility = _centered_evidence(online, end, anchor_xy)
        else:
            utility = len(gained) / len(actions)
        return dict(id=name, actions=list(map(int, actions)), source=name.split('/')[0],
                    site_xy=list(end[:2]), face_yaw=end[2], new_surfaces=len(gained),
                    surface_keys=gained, frontier_keys=boundaries,
                    geometry_score=utility, model_score=None)

    continuation = candidate(base, end, 'continue')
    candidates = [continuation]
    # At most one option per movement sector, plus an in-place turn in each
    # direction. The same compact slate is supplied to both ranking methods.
    def pose_key(p):
        return round(p[0] / .04), round(p[1] / .04), round(p[2] / TURN) % 12
    seen = {pose_key(end)}
    groups = {}
    for actions in _primitives(n):
        end = _end(online, start, actions)
        if end is None or pose_key(end) in seen:
            continue
        seen.add(pose_key(end))
        if 1 in actions:
            bearing = math.atan2(end[1] - start[1], end[0] - start[0])
            sector = round(math.remainder(bearing - start[2], math.tau) / TURN)
            group = ('step', sector)
        else:
            group = ('turn', 1 if actions[0] == 2 else -1)
        row = candidate(actions, end, group[0] + '/' + ''.join(map(str, actions)))
        if row['geometry_score'] <= 0:
            continue
        key = (row['geometry_score'], -len(actions))
        if group not in groups or key > groups[group][0]:
            groups[group] = key, row
    candidates.extend(row for key, row in groups.values())
    geometry = max(candidates, key=lambda r: r['geometry_score'])
    result.update(candidates=candidates, geometry_id=geometry['id'],
                  status='offered' if len(candidates) > 1 else 'continue_only')
    return score_observation_routes(result, scorer) if scorer is not None else result


def score_observation_routes(proposal, scorer):
    """Fork only the ranking, preserving the candidate geometry and regional tie break."""
    if len(proposal['candidates']) <= 1:
        return proposal
    candidates = [dict(row) for row in proposal['candidates']]
    scores = list(scorer([row['actions'] for row in candidates]))
    if len(scores) != len(candidates) or not all(math.isfinite(float(s)) for s in scores):
        raise ValueError('one finite model value is required for every candidate including continue')
    for row, value in zip(candidates, scores):
        row['model_score'] = float(value)
    selected = max(candidates, key=lambda row: row['model_score'])
    return dict(proposal, candidates=candidates, selected_id=selected['id'], status='scored')
