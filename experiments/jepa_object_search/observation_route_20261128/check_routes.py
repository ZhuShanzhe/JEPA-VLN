"""Deterministic checks of the observed-map proposal and frozen-score interface."""
from pathlib import Path
import math
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'local_target_20261128' / 'waypoints'))
from routes import observation_routes, score_observation_routes


class MeasuredGrid:
    res = .125

    def __init__(self):
        self.geometry_observed = np.ones((72, 72), dtype=bool)
        self.obstacle = np.zeros_like(self.geometry_observed)
        self.obstacle[47, 15:58] = True
        self.obstacle[16:50, 36] = True

    def world_to_cell(self, xy):
        return tuple(math.floor(v / self.res) for v in xy)

    def in_bounds(self, cell):
        return all(0 <= c < 72 for c in cell)

    def traversable(self, cell):
        x, y = cell
        return self.in_bounds(cell) and self.geometry_observed[y, x] and not self.obstacle[y, x]


def main():
    grid = MeasuredGrid()
    args = dict(grid=grid, pose=(4., 4., 0.), scale=1., mode='discover',
                region_actions=[2, 2, 1, 1], budget=6)
    offered = observation_routes(**args)
    options = offered['candidates']
    assert offered['status'] == 'offered' and options[0]['actions'] == [2, 2, 1, 1]
    assert all(row['new_surfaces'] > 0 for row in options[1:])
    assert len({(tuple(r['site_xy']), r['face_yaw']) for r in options}) == len(options)
    assert score_observation_routes(offered, lambda xs: [0.] * len(xs))['selected_id'] == 'continue'
    scored = score_observation_routes(offered, lambda xs: range(len(xs)))
    assert scored['selected_id'] == options[-1]['id'] and offered['candidates'][0]['model_score'] is None
    covered = {k for r in options for k in r['surface_keys']}
    # Adding all possible map surface bins disallows spurious 'discovery' by recentering.
    covered.update((x, y) for x in range(24) for y in range(24))
    assert observation_routes(**args, observed_surfaces=covered)['status'] == 'continue_only'
    args.update(mode='verify', anchor_xy=(4., 5.6))
    result = observation_routes(**args)
    assert result['status'] == 'offered'
    assert any(1 not in r['actions'] for r in result['candidates'][1:])
    grid.geometry_observed[:, :] = False
    grid.geometry_observed[32, 32] = True
    assert observation_routes(**args)['status'] == 'no_safe_continuation'
    args.update(region_actions=[2], budget=1)
    for choice in observation_routes(**args)['candidates']:
        assert 1 not in choice['actions'] and len(choice['actions']) == 1
    print('PASS: identical candidate set, return-to-region, raw action budget, viewpoint diversity, '
          'ranking-only selection, unseen-floor exclusion, and history-aware surface gating')


if __name__ == '__main__':
    main()
