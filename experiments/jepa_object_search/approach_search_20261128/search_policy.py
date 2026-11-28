"""Select a target-conditioned observation site; update search memory from executed views."""
from __future__ import annotations
from dataclasses import dataclass
import math


def xy(position):
    x, y = map(float, position)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("invalid map position")
    return x, y


def angle(heading):
    value = float(heading)
    if not math.isfinite(value):
        raise ValueError("invalid view direction")
    return math.remainder(value, math.tau)


def probability(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError("expected a probability")
    return value


@dataclass
class Observation:
    position: tuple
    heading: float
    prior: float | None


class PriorSearchMemory:
    """One target and episode, in a stable metric frame. Decisions never mark a site visited."""

    def __init__(self):
        self.views = {}
        self.inspected = set()
        self.blocked = {}
        self.tick = 0
        self.prior_peak = 0.0

    @staticmethod
    def _bin(position, heading):
        x, y = position
        return round(x / .35), round(y / .35), round(heading / (math.pi / 6)) % 12

    def mark_view(self, position, heading, visible_cells=None, observed_prior=None):
        position, heading = xy(position), angle(heading)
        self.tick += 1
        key = self._bin(position, heading)
        old = self.views.get(key)
        evidence = probability(observed_prior) if observed_prior is not None else (old.prior if old else None)
        self.views[key] = Observation(position, heading, evidence)
        if visible_cells is not None:
            self.inspected.update(visible_cells)

    def mark_blocked(self, position, heading):
        self.blocked[self._bin(xy(position), angle(heading))] = self.tick

    def update_context(self, position, heading, prior):
        key = self._bin(xy(position), angle(heading))
        if key in self.views:
            self.views[key].prior = probability(prior)

    def _view_overlap(self, position, heading):
        overlap, evidence, support = 0., 0., 0.
        for view in self.views.values():
            distance = math.dist(position, view.position)
            if distance >= 1.2:
                continue
            turn = abs(math.remainder(heading - view.heading, math.tau))
            similarity = max(0., 1. - turn / (math.pi / 2)) * (1. - distance / 1.2)
            overlap = max(overlap, similarity)
            if view.prior is not None:
                evidence += similarity * view.prior
                support += similarity
        return overlap, (evidence / support if support else None)

    def choose(self, candidates):
        if not candidates or len({c["key"] for c in candidates}) != len(candidates):
            raise ValueError("expected a nonempty list with unique candidate keys")
        rows = []
        for c in candidates:
            pos, heading = xy(c["position"]), angle(c["heading"])
            prior = probability(c["prior"])
            overlap, local_prior = self._view_overlap(pos, heading)
            if local_prior is not None:
                prior = (1. - .4 * overlap) * prior + .4 * overlap * local_prior
            sensor_cells = set(c.get("visible_cells") or ())
            unseen = 1. - overlap
            if sensor_cells:
                unseen *= 1. - len(sensor_cells & self.inspected) / len(sensor_cells)
            key = self._bin(pos, heading)
            blocked = self.tick - self.blocked.get(key, -1000) < 12
            rows.append(dict(key=c["key"], prior=prior, unseen=unseen, blocked=blocked,
                             novelty=probability(c.get("novelty", .5)), steps=max(1, int(c.get("steps", 1)))))

        peak = max(r["prior"] for r in rows)
        self.prior_peak = max(self.prior_peak, peak)
        # Relative to the episode's best available opportunity, not a calibrated chance of success.
        cutoff = .8 * self.prior_peak
        feasible = [i for i, row in enumerate(rows) if not row["blocked"]]
        if not feasible:
            feasible = list(range(len(rows)))
        fresh = [i for i in feasible if rows[i]["unseen"] >= .35]
        high = [i for i in fresh if rows[i]["prior"] >= cutoff and rows[i]["prior"] > 0.]
        phase = "prior_search" if high else "remaining_search"
        choices = high or fresh or feasible
        for row in rows:
            if high:
                # Search high-prior locations first. Break close ties by new surfaces and travel.
                value = row["prior"] * (.4 + .6 * row["unseen"])
            else:
                value = row["unseen"] * (.45 + .35 * row["prior"] + .2 * row["novelty"])
            row["score"] = value / (1. + .1 * min(row["steps"], 48) / 24)
        idx = max(choices, key=lambda i: rows[i]["score"])
        return idx, dict(phase=phase, high_cutoff=cutoff, candidates=rows)
