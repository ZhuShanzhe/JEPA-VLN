"""Episode-local correction of frontier preferences from policy-visible observations."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Entrance:
    region: str
    cue: np.ndarray


def unit(x):
    a = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(a)
    if a.ndim != 1 or not a.size or not np.isfinite(a).all() or norm < 1.e-9:
        raise ValueError("Expected a finite, nonzero vector of observable entrance features")
    return a / norm


class EpisodeEvidence:
    """One support per visited region and query. Only the executed choice may supply feedback."""
    def __init__(self, strength=0.35, minimum_coverage=0.8, score_unit=1.0):
        if not 0 <= strength <= 1 or not 0 < minimum_coverage <= 1 or not 0 < score_unit < math.inf:
            raise ValueError("Invalid correction strength or coverage threshold")
        self.strength, self.minimum_coverage, self.score_unit = strength, minimum_coverage, score_unit
        self.reset(None)

    def reset(self, episode):
        self.episode = episode
        self.pending = None
        self.support = {}

    def _check(self, episode):
        if not episode or episode != self.episode:
            raise ValueError("Reset the memory at the start of each episode")

    def scores(self, episode, query, entrances, base_scores):
        """Adjust only unvisited region IDs; the base scorer handles revisits and instance memory."""
        self._check(episode)
        base = np.asarray(base_scores, dtype=np.float64)
        if base.shape != (len(entrances),) or not len(base) or not np.isfinite(base).all():
            raise ValueError("Invalid base route scores")
        corrected = base.copy()
        # Fixed score unit, chosen on development data. Allow modest gaps to reverse even with two routes.
        scale = self.score_unit
        for i, entrance in enumerate(entrances):
            if (query, entrance.region) in self.support:
                continue
            cue = unit(entrance.cue)
            numerator, mass = 0., 1.
            for (support_query, _), (observed_cue, residual, confidence) in self.support.items():
                if support_query != query:
                    continue
                similarity = max(0., float(cue @ observed_cue)) ** 4
                weight = similarity * confidence
                numerator += weight * residual
                mass += weight
            corrected[i] += scale * self.strength * numerator / mass
        return corrected

    def begin_choice(self, episode, query, chosen, prior_presence=None):
        """Called after the host confirms this route is being executed."""
        self._check(episode)
        if self.pending is not None:
            raise RuntimeError("Finish or cancel the previous choice before recording another")
        if not chosen.region or not query:
            raise ValueError("Missing online region ID or query")
        # A gain ranker's logits do not express the probability that a room contains the goal.
        p = 0.5 if prior_presence is None else float(prior_presence)
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Presence prior must be a calibrated probability or None")
        self.pending = (query, chosen.region, unit(chosen.cue), p)

    def cancel_choice(self, episode):
        self._check(episode)
        self.pending = None

    def finish_choice(self, episode, *, arrived, target_seen, coverage, confidence):
        """All fields come from the robot; coverage is the assessed inspected fraction."""
        self._check(episode)
        if self.pending is None:
            raise RuntimeError("No executed choice awaiting feedback")
        if type(arrived) is not bool or type(target_seen) is not bool:
            raise ValueError("Expected explicit boolean outcomes")
        if not (math.isfinite(coverage) and 0 <= coverage <= 1 and
                math.isfinite(confidence) and 0 <= confidence <= 1):
            raise ValueError("Invalid observation reliability")
        query, region, cue, prior = self.pending
        self.pending = None
        if not arrived or confidence == 0 or (not target_seen and coverage < self.minimum_coverage):
            return False
        # Positive evidence needs a reliable detection. Negative evidence additionally needs coverage.
        weight = confidence if target_seen else confidence * coverage
        key = (query, region)
        old = self.support.get(key)
        if old:
            _, old_residual, old_weight = old
            # A second look that simply misses a positively observed object is not contradictory proof.
            if not target_seen and (old_residual > 0 or old_weight >= weight):
                return False
            if target_seen and old_residual > 0 and old_weight >= weight:
                return False
        self.support[key] = (cue, float(target_seen) - prior, weight)
        return True


def controlled_check():
    """Deterministic paired-house interface regression. No claim of learned navigation ability."""
    memory = EpisodeEvidence()
    a = np.array([1., 0.])
    b = np.array([0., 1.])
    unseen = [Entrance("next-a", a), Entrance("next-b", b)]
    matched = independent = 0
    for support_type in (0, 1):
        episode = f"fixture-{support_type}"
        memory.reset(episode)
        assert np.array_equal(memory.scores(episode, "chair", unseen, [0,0]), [0,0])
        memory.begin_choice(episode, "chair", Entrance("opened", (a,b)[support_type]))
        assert memory.finish_choice(episode, arrived=True, target_seen=True, coverage=.1, confidence=.9)
        weak_wrong_prior = [0.02,0.] if support_type else [0.,0.02]
        scores = memory.scores(episode, "chair", unseen, weak_wrong_prior)
        selected = int(np.argmax(scores))
        matched += selected == support_type
        independent += sum(selected == placement for placement in (0, 1))
        assert np.array_equal(memory.scores(episode, "monitor", unseen, [0,0]), [0,0])
        memory.begin_choice(episode, "chair", Entrance("blocked", (b,a)[support_type]))
        assert not memory.finish_choice(episode, arrived=False, target_seen=False, coverage=1., confidence=.9)
        memory.begin_choice(episode, "chair", Entrance("glimpse", (b,a)[support_type]))
        assert not memory.finish_choice(episode, arrived=True, target_seen=False, coverage=.1, confidence=.9)
        assert np.array_equal(scores, memory.scores(episode, "chair", unseen, weak_wrong_prior))
    assert matched == 2 and independent == 2
    return {"scope":"synthetic region-cue contracts", "same_rule_correct": matched, "same_rule_total":2,
            "independent_placement_correct":independent, "independent_placement_total":4,
            "no_evidence_tie":True, "unknown_stays_unknown":True, "reset_per_episode":True}


if __name__ == "__main__":
    import json
    print(json.dumps(controlled_check()))
