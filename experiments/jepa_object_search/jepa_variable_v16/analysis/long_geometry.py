"""Check archived local-map geometry for distant forks sharing a whole forecast window."""
from collections import Counter
import json
import os
from pathlib import Path
import sys

from coverage import digest, events, read_json


def main(project):
    sys.path.insert(0, str(project / "jepa_variable_v16/adapter"))
    from agents.navigator import PathFollower
    from delayed_forks import follow, trunk_plan
    from variable_routes import build_table
    reach = float(os.getenv("NAV_REACH_M", ".8"))
    follower = PathFollower(scale=1., reach_m=reach)
    pose = (0.,0.,0.)
    totals = {}
    for name in ("jepa_evidence_v15_first2", "jepa_round13_horizon_five"):
        pilot = project / "mapping_runtime/pilots" / name
        bins = Counter()
        for row in read_json(pilot/"panel.json")["rows"]:
            ref = read_json(pilot/"comparison/accepted/second"/(row["episode_id"]+".json"))
            result = Path(ref["result"])
            assert digest(result) == ref["result_sha256"]
            limit = read_json(result)["episodes"][0]["max_steps_budget"]
            directory = result.parent / "agent"
            checks = events(directory/"v13_routes.jsonl","eligibility")
            probes = events(directory/"frontier_probe.jsonl","activation")
            for check in checks:
                bins["checks"] += 1
                if check["reason"] in ("native_action_not_explore","off_level","unsupported_target_category"):
                    bins["other_guard"] += 1
                    continue
                activation, = [a for a in probes if a["activated"] and a["step"]==check["step"] and
                               a["source"]==check["source"] and a["selected_id"]==check["original"]]
                rows = activation["candidates"]
                budget = limit - check["step"] - 4
                _, reason = build_table(rows, check["original"], budget)
                if reason is None:
                    bins["immediate"] += 1
                    continue
                if reason != "no_distinct_spatial_branch_in_horizon" or budget <= 24:
                    bins["other_local_fallback"] += 1
                    continue
                clusters = []
                for i, entry in enumerate(rows):
                    assert entry["path_forward_left_m"] is not None
                    path = entry["path_forward_left_m"]
                    forecast = follow(PathFollower, path, pose, 1., reach, 24)
                    assert forecast == entry["nominal_action_ids"]
                    # Per-event keys check geometry only; live stale-identity checks
                    # require the current production map and are not reconstructed.
                    clusters.append(dict(key=(i,),world=entry["endpoint_forward_left_m"],scale=1.,path=path))
                i = next(i for i,r in enumerate(rows) if r["id"] == check["original"])
                delayed = trunk_plan(clusters,i,pose,follower,budget)
                bins["no_nearby_fork_but_distinct_distant_geometry" if delayed else "no_distant_split_detected"] += 1
        totals[name] = dict(bins)
    return dict(scope="Geometric screen on native plans from accepted archived decisions. "
                      "Live map refresh and high-priority interrupts cannot be reconstructed by this screen.",
                panels=totals)


if __name__ == "__main__":
    report = main(Path(sys.argv[1]).resolve())
    Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))
