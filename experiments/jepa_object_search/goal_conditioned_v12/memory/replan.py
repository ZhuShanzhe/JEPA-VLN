"""Rerank uninspected plans using actual, query-conditioned evidence from one probe."""
import math

import numpy as np


def poses(actions, collisions):
    pos = np.zeros(2)
    yaw = 0.
    xy, directions = [], []
    for action, blocked in zip(actions, collisions):
        if action == 0 and not blocked:
            pos = pos + .25 * np.array([math.cos(yaw), math.sin(yaw)])
        if action == 1:
            yaw += math.pi/6
        if action == 2:
            yaw -= math.pi/6
        xy.append(pos.copy())
        directions.append([math.cos(yaw), math.sin(yaw)])
    return np.array(xy), np.array(directions)


def evidence_features(ctx, text, plans, first, seen, seen_cue, start_cue, collision, recent_memory):
    """Inputs include the selected trajectory only. No outcome or unselected frames."""
    from train_compare import route_inputs
    D = len(text)
    inp = route_inputs(ctx, plans)
    xy, direction = poses(plans[first], collision)
    ix = np.arange(5, len(xy), 6)
    locations = xy[ix]
    eye = direction[ix]
    result = []
    for j, plan in enumerate(plans):
        endpoint, hd = poses(plan, np.zeros(len(plan), bool))
        near = np.exp(-np.linalg.norm(endpoint[ix,None]-locations[None],axis=-1) / 1.5)
        overlap = near.max(1)
        aligned = eye @ hd[ix].T
        entry = inp[j,:,:D]
        start_match = entry @ text
        observation = seen[:,0,:] @ entry.T
        observed_match = (seen @ text).max(-1)
        adjusted = seen_cue - np.mean(start_cue)
        # Associate each visited clue with candidate-specific appearance and distance.
        soft = np.exp(2 * observation) * near.T
        weighted = np.average(adjusted, weights=soft.mean(1).clip(1e-6))
        terminal = endpoint[-1]
        distances = np.linalg.norm(locations-terminal,axis=1)
        result.append([*recent_memory[j], np.linalg.norm(terminal)/6,
            np.mean(overlap), overlap[-1], (distances.min()/6), np.mean(aligned),
            np.mean(start_match),np.max(start_match),start_match[-1],
            np.mean(observation),np.max(observation),np.mean(observation[-1]),
            weighted, np.mean(near @ adjusted), np.mean(observation.T @ adjusted),
            np.mean(near @ observed_match), np.mean(observation.T @ observed_match),
            np.mean(overlap)*np.max(seen_cue), np.mean(overlap)*np.mean(collision),
            start_match.max()*np.max(adjusted), np.mean(observation)*np.max(adjusted)])
    return np.asarray(result,float)


def second_scores(initial_scores, initial_bonus, updated_bonus, features, learned, first):
    residual = (features / learned['scale']) @ learned['coef']
    centered = residual - residual[[i for i in range(len(features)) if i != first]].mean()
    corrected = initial_scores - initial_bonus + updated_bonus + learned['gate'] * np.clip(centered, -1., 1.)
    corrected[first] = -np.inf
    return corrected
