"""Find the first chronological difference between matched complete episodes."""
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_bytes())


def first_step(events, kind, allowed=lambda r: True):
    return next((r["step"] for r in events if r["event"] == kind and allowed(r)), None)


def compute(pilot):
    output = []
    for row in read(pilot / "panel.json")["rows"]:
        ref = pilot / "comparison/accepted/second" / (row["episode_id"] + ".json")
        if not ref.exists():
            continue
        ref_a = read(Path(row["ref"]))
        ref_b = read(ref)
        assert ref_a["episode_id"] == ref_b["episode_id"] == row["episode_id"]
        a, = read(Path(ref_a["result"]))["episodes"]
        target_path = Path(ref_b["result"])
        b, = read(target_path)["episodes"]
        ra, rb = a["paper"]["steps"], b["paper"]["steps"]
        assert a["goal_text"] == b["goal_text"] and a["scene_id"] == b["scene_id"]
        assert ra[0]["position_before"] == rb[0]["position_before"]
        limit = min(len(ra), len(rb))
        different = next((n for n in range(limit) if
                          (ra[n]["action"], ra[n]["action_role"]) != (rb[n]["action"], rb[n]["action_role"])),
                         limit if len(ra) != len(rb) else None)
        route = [json.loads(s) for s in (target_path.parent / "agent/v13_routes.jsonl").read_bytes().splitlines() if s]
        scan = first_step(route, "scan_start")
        rank = first_step(route, "rank")
        other = first_step(route, "rank", lambda e: e["selected"] != e["original"])
        def count_reports_before(ep, t):
            return sum(len(r["new_targets"]) for r in ep["report_events"] if t is None or r["step"] < t)
        output.append(dict(ordinal=row["ordinal"], common_action_prefix=len(ra) if different is None else different,
                           first_differing_action_step=different, first_scan_start_step=scan, first_rank_step=rank,
                           first_alternative_selection_step=other,
                           first_difference_predates_any_scan=different is not None and (scan is None or different < scan),
                           credited_by_treatment_before_its_first_scan=count_reports_before(b, scan),
                           credited_by_native_before_the_same_step=count_reports_before(a, scan)))
    return {"cases": output}


if __name__ == "__main__":
    pilot, dest = map(Path, sys.argv[1:])
    dest.write_text(json.dumps(compute(pilot), indent=2))
