"""Score each frontier in common six-action evidence units, without set normalization."""
import numpy as np


def score_routes(memory_score, evidence_rate, supply, deficit, remaining):
    memory_score, evidence_rate, supply = (np.asarray(x, dtype=float)
                                          for x in (memory_score, evidence_rate, supply))
    if (memory_score.ndim != 1 or len(memory_score) < 2 or
            memory_score.shape != evidence_rate.shape or memory_score.shape != supply.shape or
            not np.isfinite([memory_score, evidence_rate, supply]).all() or
            not np.all((evidence_rate >= 0) & (evidence_rate <= 1)) or
            not np.all((supply >= 0) & (supply <= 1)) or not 0 <= deficit <= 1 or remaining < 0):
        raise ValueError("invalid evidence, memory, coverage or budget")
    availability = min(1., remaining / 48.)
    weight = .5 * deficit * availability
    evidence = (.25 + .75 * deficit) * evidence_rate
    novelty = weight * supply * (.35 + .65 * evidence_rate)
    final = memory_score + evidence + novelty
    return dict(index=int(np.argmax(final)), final_scores=final.tolist(),
                memory_scores=memory_score.tolist(), evidence_bonus=evidence.tolist(),
                exploration_bonus=novelty.tolist(), unvisited_fraction=supply.tolist(),
                no_exploration_index=int(np.argmax(memory_score + evidence)),
                no_evidence_index=int(np.argmax(memory_score + weight * supply * .35)),
                clue_deficit=float(deficit), adaptive_weight=float(weight), budget_factor=float(availability))
