"""Teacher for count-of-new-reportable-instance labels in a navigable indoor scene."""
from collections import defaultdict

import numpy as np


class ReportViewpoints:
    """Index independent instance viewpoints on the same navmesh as the candidate rollouts."""

    def __init__(self, pathfinder, goals, radius=.25):
        import habitat_sim
        from scipy.spatial import cKDTree

        self.pf, self.hs, self.radius = pathfinder, habitat_sim, radius
        self.ids = np.asarray([int(g["object_id"]) for g in goals])
        if len(set(self.ids)) != len(goals) or not goals:
            raise ValueError("Expected a nonempty catalogue with unique scene instance IDs")
        coords, owners = [], []
        for i, goal in enumerate(goals):
            snaps = [self.pf.snap_point(v["agent_state"]["position"]) for v in goal["view_points"]]
            if not snaps:
                raise ValueError("Instance has no independent viewpoint")
            snaps = np.asarray(snaps, np.float32)
            snaps = np.unique(snaps[np.isfinite(snaps).all(axis=1)], axis=0)
            if not len(snaps):
                raise ValueError("Instance has no navigable viewpoint")
            coords.extend(snaps)
            owners.extend([i] * len(snaps))
        self.coords = np.asarray(coords, np.float32)
        self.owners = np.asarray(owners)
        self.tree = cKDTree(self.coords)

    def eligible(self, position):
        """Euclidean distance only filters points. Each positive requires an actual geodesic path."""
        start = np.asarray(self.pf.snap_point(position), dtype=np.float32)
        if not np.isfinite(start).all():
            raise ValueError("Position is off the navmesh")
        candidate = defaultdict(list)
        for i in self.tree.query_ball_point(start, self.radius + 1e-5):
            candidate[self.owners[i]].append(i)
        reached = np.zeros(len(self.ids), dtype=bool)
        for index, nearest in candidate.items():
            path = self.hs.MultiGoalShortestPath()
            path.requested_start = start
            path.requested_ends = self.coords[nearest].tolist()
            if self.pf.find_path(path):
                reached[index] = path.geodesic_distance <= self.radius
        return reached


def new_instance_counts(hits, eligible_at_start, instance_ids, target_mask, reported_ids=()):
    """Return one integer count per route. Repeated visits and repeated reports earn no extra credit."""
    hits = np.asarray(hits, dtype=bool)
    start = np.asarray(eligible_at_start, dtype=bool)
    ids = np.asarray(instance_ids)
    target = np.asarray(target_mask, dtype=bool)
    if hits.ndim != 3 or start.shape != ids.shape or target.shape != ids.shape or hits.shape[2:] != ids.shape:
        raise ValueError("Expected [routes, steps, instances] hits and matching instance vectors")
    allowed = target & ~start & ~np.isin(ids, list(reported_ids))
    return (hits.any(axis=1) & allowed).sum(axis=1)
