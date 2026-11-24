"""Offline Habitat physics and RGB checks on archived distant fork geometries."""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def cases(project):
    from agents.navigator import PathFollower
    from delayed_forks import follow, trunk_plan
    from variable_routes import build_table
    from verify_habitat import digest, read, rows
    reach = float(os.getenv("NAV_REACH_M", ".8"))
    folder = project / "mapping_runtime/pilots/jepa_round13_horizon_five"
    for panel in read(folder / "panel.json")["rows"]:
        ref = read(folder / "comparison/accepted/second" / (panel["episode_id"] + ".json"))
        result = Path(ref["result"])
        assert digest(result) == ref["result_sha256"]
        document = read(result)
        ep, = document["episodes"]
        logs = result.parent / "agent"
        activations = rows(logs / "frontier_probe.jsonl")
        for gate in rows(logs / "v13_routes.jsonl"):
            if gate["event"] != "eligibility" or gate["reason"] in (
                    "native_action_not_explore", "off_level", "unsupported_target_category"):
                continue
            step = gate["step"]
            item, = [x for x in activations if x["event"] == "activation" and x["activated"] and
                     x["step"] == step and x["source"] == gate["source"] and
                     x["selected_id"] == gate["original"]]
            paths = item["candidates"]
            budget = ep["max_steps_budget"] - step - 4
            _, reason = build_table(paths, gate["original"], budget)
            if reason != "no_distinct_spatial_branch_in_horizon" or budget <= 24:
                continue
            clusters = []
            for i, p in enumerate(paths):
                path = p["path_forward_left_m"]
                assert follow(PathFollower, path, (0, 0, 0), 1., reach, 24) == p["nominal_action_ids"]
                clusters.append(dict(key=(i,), path=path, world=p["endpoint_forward_left_m"], scale=1.))
            i = next(i for i, p in enumerate(paths) if p["id"] == gate["original"])
            delayed = trunk_plan(clusters, i, (0, 0, 0), PathFollower(scale=1., reach_m=reach), budget)
            if delayed is not None:
                yield panel["ordinal"], document, step, delayed[0]


def rgb(frame):
    return np.asarray(frame["rgb"])[..., :3].astype(np.int16)


def probe(document, step, prefix):
    from evaluation.main.evaluator import _env_action
    from verify_habitat import difference, replay_until, scene, state
    env, item, dataset = scene(document)
    timeline = document["episodes"][0]["paper"]["steps"]
    try:
        first = replay_until(env, item, dataset, timeline, step)
        origin = state(env)
        path_length = 0.
        reached = 0
        collision = False
        native_agreement = 0
        for i, action in enumerate(prefix):
            if (native_agreement == i and step + i < len(timeline) and
                    timeline[step + i]["action_role"] == "navigate" and
                    timeline[step + i]["action"] == action):
                native_agreement += 1
            old = state(env)
            frame = env.step(_env_action(action))
            d, _ = difference(old, state(env))
            path_length += d
            collision = action == 1 and bool(env.sim.previous_step_collided)
            if collision:
                break
            reached += 1
        arrival = state(env)
        displacement, heading = difference(origin, arrival)
        record = dict(required_common_steps=len(prefix), observed_without_collision=reached,
                      archive_matching_prefix=native_agreement, collision=collision,
                      distance_traveled_m=path_length, displacement_m=displacement,
                      heading_change_deg=heading, rotated_scan_completed=False)
        if not collision and reached == len(prefix) and displacement >= .5:
            view_deltas = []
            frames = []
            for i, action in enumerate((3, 2, 2, 3)):
                frame = env.step(_env_action(action))
                if i < 3:
                    img = rgb(frame)
                    frames.append(img)
                    view_deltas.append(float(np.abs(img - rgb(first)).mean()))
            restoration = difference(arrival, state(env))
            assert restoration[0] < .002 and restoration[1] < .05
            assert all(not np.array_equal(frames[i], frames[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
            record.update(rotated_scan_completed=True, distinct_views=len(frames),
                          mean_rgb_delta_from_origin=view_deltas,
                          scan_position_error_m=restoration[0], scan_heading_error_deg=restoration[1])
        return record
    finally:
        env.close()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("project", type=Path)
    args.add_argument("output", type=Path)
    a = args.parse_args()
    if a.output.exists():
        args.error("output must be a fresh file")
    from verify_rank import deny_ip_sockets
    deny_ip_sockets()
    results = []
    for ordinal, document, step, prefix in cases(a.project):
        report = dict(ordinal=ordinal, step=step, **probe(document, step, prefix))
        results.append(report)
        print(json.dumps(report), flush=True)
    assert len(results) == 2
    summary = dict(archived_distant_forks=len(results), ip_sockets_denied=True, cases=results,
                   scope="Archived native geometry, physical Habitat motion and fresh RGB at the "
                         "reached viewpoint. Historical live A* revisions and VLM interrupts are unavailable. "
                         "No full-episode detection or search outcome is assessed.")
    with a.output.open("x") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")
