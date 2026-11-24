"""Episode-local evidence gate. Inputs available to the navigation policy only."""
import math
import numpy as np


def claim_rays(memory, trace, report_steps):
    # Reproduce this worker's existing dead-reckoning convention. A claim masks
    # only the observation ray used to report; it does not confirm an instance.
    reports = set(report_steps)
    if not reports:
        return set()
    xy, yaw = memory.xy.copy(), memory.yaw
    rays = set()
    for t, action, blocked, _ in trace:
        if action == 0 and not blocked:
            xy += .25 * np.array([math.cos(yaw), math.sin(yaw)])
        elif action == 1:
            yaw += math.pi/6
        elif action == 2:
            yaw -= math.pi/6
        if t in reports:
            rays.add((memory.cell(xy), int(round(yaw/(math.pi/3))) % 6))
    return rays


def clue_deficit(memory, claimed, last_report):
    cal = memory.cue_model
    background = float(cal["train_first_appearance_frequency"])
    upper = float(max(cal["evidence"]))
    if upper <= background:
        raise ValueError("invalid evidence calibration")
    strength = 0.
    used = 0
    for ray, (step, _, p) in memory.cells.items():
        if ray in claimed or step <= last_report:
            continue
        relative = np.clip((p-background)/(upper-background), 0., 1.)
        # 24 actual actions is the native forecast horizon. Fresh strong cues
        # suppress scouting; stale observations do not permanently lock a room.
        strength = max(strength, float(relative) * math.exp(-(memory.time-step)/24.))
        used += 1
    return 1.-strength, used


def score_routes(base, p, supply, deficit, remaining):
    base, p, supply = (np.asarray(values, dtype=float) for values in (base, p, supply))
    if (base.shape != (3,) or p.shape != base.shape or supply.shape != base.shape
            or not np.isfinite([base, p, supply]).all() or np.any((p < 0) | (p > 1))
            or np.any((supply < 0) | (supply > 1)) or not 0 <= deficit <= 1 or remaining < 0):
        raise ValueError("invalid route scores, evidence, novelty or budget")
    # One term rewards predicted new target evidence. The other pays for
    # previously unvisited route cells when there are few fresh target clues.
    availability = min(1., remaining / 48.)
    w = .5 * deficit * availability
    evidence = (.25 + .75 * deficit) * p
    novelty = w * supply * (.35 + .65*p)
    scores = base + evidence + novelty
    return dict(index=int(np.argmax(scores)), old_base_index=int(np.argmax(base)),
                no_exploration_index=int(np.argmax(base + evidence)),
                no_evidence_index=int(np.argmax(base + w * supply * .35)),
                final_scores=scores.tolist(), evidence_probability=p.tolist(),
                evidence_bonus=evidence.tolist(), exploration_bonus=novelty.tolist(),
                unvisited_fraction=supply.tolist(), clue_deficit=float(deficit),
                adaptive_weight=float(w), budget_factor=float(availability))
