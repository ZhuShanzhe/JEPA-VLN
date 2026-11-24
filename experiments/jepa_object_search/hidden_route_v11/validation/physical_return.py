"""Continuously execute a first route and return, then test both alternatives at the reached pose."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


def atomic_npz(path, **data):
    tmp = path.with_suffix(".pending.npz")
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)


def acquire(obs, lookup):
    ids, depth = obs["ids"], obs["depth"]
    active = (ids > 0) & (ids < len(lookup)) & np.isfinite(depth) & (depth > 0) & (depth <= 2.)
    counts = np.bincount(ids[active], minlength=len(lookup))
    return set(map(int, np.flatnonzero(counts >= 32)))


def search_leg(sim, actions, lookup, commands):
    agent = sim.get_agent(0)
    result, captures = set(), []
    for t, command in enumerate(actions, 1):
        agent.act(commands[int(command)])
        if t % 6 == 0:
            obs = sim.get_sensor_observations()
            captures.append(obs["rgb"][:, :, :3].copy())
            result.update(acquire(obs, lookup))
    return result, np.array(captures)


def reverse_route(actions):
    # Reverse the complete nominal route using only issued commands, including any
    # unsuccessful forwards. Every return therefore uses exactly 36 actions.
    opposite = (0, 2, 1)
    return [1] * 6 + [opposite[int(a)] for a in actions[::-1]] + [1] * 6


def distance_images(a, b):
    return float(np.mean(np.abs(a.astype("int16") - b.astype("int16"))))


def angle_error(a, b):
    qa, qb = (np.array([q.real, *q.imag], dtype=float) for q in (a, b))
    dot = np.clip(abs(np.dot(qa, qb))/(np.linalg.norm(qa)*np.linalg.norm(qb)), 0., 1.)
    return math.degrees(2. * math.acos(dot))


def run(project, asset, house):
    sys.path.insert(0, str(project / "hssd"))
    from collect_branches import ACT, WORDS, semantics, simulate, world_state

    source = project / "hssd/test" / house
    out, teacher = (project / "validation" / k / house for k in ("returns", "returns_truth"))
    out.mkdir(parents=True, exist_ok=True)
    teacher.mkdir(parents=True, exist_ok=True)
    index = json.loads((source / "private_replay.json").read_bytes())
    complete = out / "receipt.json"
    if complete.exists():
        return
    sim = simulate(asset, house)
    stats = dict(origins=0, branches=0, successful_returns=0, alternatives_from_return=0,
                 exact_first_gain=True, max_origin_rgb_mae=0., max_first_frame_mae=0.,
                 max_return_position_error_m=0., max_return_angle_error_deg=0., max_return_rgb_mae=0.)
    try:
        lookup = semantics(sim, asset)
        agent = sim.get_agent(0)
        for entry in index:
            uid = entry["origin"]
            base = world_state(entry["position"], entry["yaw"])
            with np.load(source / (uid + ".npz")) as archive:
                actions, expected = archive["actions"], archive["y"]
                words, old_frames, context = archive["words"], archive["future"], archive["ctx"]
            n, q = expected.shape
            assert n == 3 and actions.shape == (3, 24)
            word_idx = np.array([WORDS.index(w) for w in words])
            def classify(found):
                return np.array([sum(lookup[i] == target for i in found) for target in word_idx])
            home_actions, home_collisions, home_rgb, valid = [], [], [], []
            second_y = np.full((n,n,q), -1, dtype=int)
            union_y = second_y.copy()
            for i in range(n):
                # Re-initialization separates counterfactual histories only. There is
                # no teleport from a first route's endpoint back to the junction.
                agent.set_state(base)
                first_img = sim.get_sensor_observations()["rgb"][:, :, :3]
                diff0 = distance_images(first_img, context[1])
                stats["max_origin_rgb_mae"] = max(stats["max_origin_rgb_mae"], diff0)
                if diff0 > 2.:
                    raise ValueError(f"initial observation mismatch for {uid}")
                seen, frames = search_leg(sim, actions[i], lookup, ACT)
                first_error = max(distance_images(x,y) for x,y in zip(frames, old_frames[i]))
                stats["max_first_frame_mae"] = max(stats["max_first_frame_mae"], first_error)
                if first_error > 2. or not np.array_equal(classify(seen), expected[i]):
                    raise ValueError(f"first route replay mismatch for {uid}/{i}")
                inverse, collisions, rgb = reverse_route(actions[i]), [], []
                for t, command in enumerate(inverse, 1):
                    feedback = agent.act(ACT[command])
                    collisions.append(bool(feedback) if command == 0 else False)
                    if t % 6 == 0:
                        rgb.append(sim.get_sensor_observations()["rgb"][:, :, :3].copy())
                reached = agent.get_state()
                pos_error = float(np.linalg.norm(reached.position - base.position))
                rot_error = angle_error(reached.rotation, base.rotation)
                rgb_error = distance_images(rgb[-1], context[1])
                ok = pos_error <= .05 and rot_error <= 1. and rgb_error <= 5.
                for name,v in [("max_return_position_error_m",pos_error),("max_return_angle_error_deg",rot_error),
                               ("max_return_rgb_mae",rgb_error)]:
                    stats[name] = max(stats[name], v)
                home_actions.append(inverse); home_collisions.append(collisions); home_rgb.append(rgb); valid.append(ok)
                stats["branches"] += 1
                stats["successful_returns"] += int(ok)
                if not ok:
                    continue
                alternatives = [j for j in range(n) if j != i]
                for k,j in enumerate(alternatives):
                    # The second hypothetical continuation forks from the captured
                    # physically reached state (including its residual position error).
                    if k:
                        agent.set_state(reached)
                    again, _ = search_leg(sim, actions[j], lookup, ACT)
                    second_y[i,j], union_y[i,j] = classify(again), classify(seen | again)
                    stats["alternatives_from_return"] += 1
            atomic_npz(teacher / (uid + ".npz"), first_y=expected, second_y=second_y, union_y=union_y,
                       words=words, origin=uid)
            atomic_npz(out / (uid + ".npz"), return_actions=np.asarray(home_actions, np.uint8),
                       return_collided=np.asarray(home_collisions, bool), return_rgb=np.asarray(home_rgb, np.uint8),
                       return_valid=np.asarray(valid, bool), origin=uid)
            stats["origins"] += 1
    finally:
        sim.close()
    stats["source_replay_index_sha256"] = hashlib.sha256((source / "private_replay.json").read_bytes()).hexdigest()
    t = complete.with_suffix(".pending")
    t.write_text(json.dumps(stats, indent=2)); t.replace(complete)
    print(house, stats, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("project", type=Path)
    p.add_argument("asset", type=Path)
    p.add_argument("house")
    args = p.parse_args()
    run(args.project, args.asset, args.house)
