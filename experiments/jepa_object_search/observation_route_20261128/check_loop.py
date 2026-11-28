"""Physical, 300-action discovery interface check. The same upper intent is held fixed in both arms."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "multiview_nav_20261128"))
from run_multiview import BoundPolicy, FAMILY, ScanRecorder, environment, old, physical, read
from multi_policy import MultiViewPolicy
from multi_worker import start_worker
from policy import ObservationRoutePolicy
from host_dispatch import exact_physical_follow


class AlwaysExplore:
    def choose(self, **inputs):
        return dict(mode="explore", region_id=inputs["regions"][0]["id"],
                    cue=None, patch_id=None, target_evidence="none")


class NoBox:
    def select(self, *_):
        raise AssertionError("the explore-only interface requested an object box")


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def receipt(events, trajectory, row):
    moves = {"move_forward": 1, "turn_left": 2, "turn_right": 3}
    counts = Counter()
    outcomes = Counter()
    for event in events:
        if "action_source" not in event:
            continue
        assert event.get("gpt_mode") in ("explore", None) and not event.get("cue_selected")
        if event.get("gpt_called"):
            assert event["old_selected_region_key"] == event["region_key"]
        outcome = event.get("gateway_reason") or event.get("exec_status")
        if outcome:
            outcomes[outcome] += 1
        if event["action_source"] != "local":
            continue
        begin, end = event["step"], event["last_step"]
        actual = [moves[t["action"]] for t in trajectory[begin+1:end+1]]
        plan = list(event["actions"])
        assert plan and len(plan) <= 4 and actual == plan[:len(actual)]
        assert len(actual) == len(plan) or event["contact"] or end == 300
        counts["local_dispatches"] += 1
        counts["primitives_verified"] += len(actual)
        counts["local_contacts"] += int(event["contact"])
    assert row["steps"] == 300 and len(trajectory) == 301
    assert counts["primitives_verified"] == row["stats"].get("local_steps", 0)
    return dict(steps=300, local_dispatches=counts["local_dispatches"],
                executed_and_verified=counts["primitives_verified"],
                local_contacts=counts["local_contacts"], route_outcomes=dict(outcomes),
                scan_steps=row["stats"].get("scan_steps", 0),
                region_steps=row["stats"].get("region_steps", 0),
                collisions=row["stats"].get("collisions", 0),
                distinct_approached=row["distinct_approached"], goal=row["goal"])


def run():
    os.umask(0o077)
    private = HERE / "private_loop"
    private.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="attempt_", dir=private))
    panel = read(FAMILY / "panel.json")
    channel, worker = start_worker(ROOT, root / "worker.log", cpu=True)
    rows = []
    try:
        for index in (0, 1):
            for arm, cls in (("regional", MultiViewPolicy), ("combined", ObservationRoutePolicy)):
                ep, sim = environment(panel[index])
                try:
                    # No service client is constructed by this runner.
                    policy = cls(AlwaysExplore(), cue=NoBox())
                    scans = ScanRecorder(sim)
                    stem = root / f"{index}_{arm}"
                    events, trajectory = stem.with_suffix(".events"), stem.with_suffix(".poses")
                    with events.open("w") as log, trajectory.open("w") as trace, exact_physical_follow(physical):
                        result = old.rollout(scans, ep, panel[index], channel, BoundPolicy(policy, scans), log, trace)
                    row = receipt(json_lines(events), json_lines(trajectory), result)
                    row.update(house_index=index, arm=arm)
                    rows.append(row)
                    print(index, arm, row, flush=True)
                finally:
                    sim.close()
    finally:
        channel.close()
        worker.wait(timeout=60)
    assert len(rows) == 4 and all(r["local_dispatches"] == 0 for r in rows if r["arm"] == "regional")
    files = ("check_loop.py", "policy.py", "gateway.py", "routes.py", "host_dispatch.py")
    return dict(fixed_house_indices=[0,1], task="discovery interface with scripted EXPLORE",
                budget_per_arm_house=300, paid_calls=0, results=rows,
                source_sha256={p: sha256((HERE/p).read_bytes()).hexdigest() for p in files})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
