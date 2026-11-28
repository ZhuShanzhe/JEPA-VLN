"""Map-only counterexample: how many 'new' surface bins were already in the scan?

Both scans use the same synthetic measured corner and robot state. The deployed
proposal traces nine rays per view. The audit traces every 0.25 degree across
the three cameras' combined 150-degree horizontal FOV. This is a controlled
geometry check, not navigation ground truth or a measure of detection recall.
"""
import json
import math
from check_routes import MeasuredGrid, observation_routes
from routes import _Map


def check():
    grid = MeasuredGrid()
    pose = (4., 4., 0.)
    online = _Map(grid, 1.)
    initial = set()
    for i in range(-300, 301):
        hit = online.ray_hit(pose[:2], math.radians(i / 4), 3.)
        if hit is not None:
            initial.add(tuple(round(x / .4) for x in hit))
    proposal = observation_routes(grid, pose, 1., mode='discover',
                                  region_actions=[2, 2, 1, 1], budget=4)
    rows = []
    for option in proposal['candidates']:
        if option['id'] == 'continue':
            continue
        predicted = set(map(tuple, option['surface_keys']))
        rows.append(dict(candidate=option['id'], predicted_new=len(predicted),
                         already_in_dense_start=len(predicted & initial)))
    n = sum(r['predicted_new'] == r['already_in_dense_start'] for r in rows)
    assert (len(rows), n) == (8, 3)
    return dict(local_candidates=len(rows), all_claimed_bins_already_visible=n,
                densely_visible_initial_bins=len(initial), rows=rows)


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
