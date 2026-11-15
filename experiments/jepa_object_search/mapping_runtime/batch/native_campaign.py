"""Run the frozen ManyON baseline with the same eligibility and provenance checks as the paired runner."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import paired as p


AGENT = "agents.nav_agent:NavAgent"


def lock_inputs():
    # Collect shared files through the paired runner's existing hash inventory.
    source = p.REPO / "agents/nav_agent.py"
    files = p.pinned_files(dict(adapter_root=str(p.REPO), source=str(source),
                                config=[], checkpoints=[], environment={}))
    return dict(namespace="formal_frozen", schema=1, arms=["native"], native_agent=AGENT,
                policy_commit=subprocess.check_output(["git", "-C", str(p.REPO), "rev-parse", "HEAD"],
                                                      text=True).strip(),
                model=p.MODEL, panel_sha256=p.PANEL_HASH, episodes=p.panel(),
                protocol=dict(action_protocol="goat", threshold=0.25, goal_type="description",
                              instruction_policy="first", seed=0, budget=300),
                weights_receipt_sha256=p.digest(p.RT / "weights/verified_weights.json"),
                files=files, scene_files=p.scene_inventory())


def log(message, **fields):
    print(json.dumps(dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          event=message, **fields)), flush=True)


def run(out):
    os.umask(0o077)
    out.mkdir(parents=True, exist_ok=True)
    with p.mutex(out / "controller.lock"):
        lock = lock_inputs()
        p.verify_weights()
        path = out / "lock.json"
        if path.exists():
            p.require(p.data(path) == lock, "Campaign hash mismatch")
        else:
            p.write(path, lock)
        satisfied = []
        for row in lock["episodes"]:
            if (out / "PAUSE").exists():
                log("paused", accepted=len(satisfied))
                return
            episode = p.accepted(out, lock, "native", row)
            if episode is None:
                log("start", ordinal=len(satisfied)+1, condition=row["condition"])
                p.run_one(out, lock, "native", row)
                episode = p.accepted(out, lock, "native", row)
                p.require(episode is not None, "Result missing immediately after run")
                log("accepted", ordinal=len(satisfied)+1, condition=row["condition"])
            satisfied.append((row["condition"], episode))
        groups = sorted({condition for condition, _ in satisfied})
        summary = {g: p.aggregate_metrics([e for c, e in satisfied if c == g]) for g in groups}
        summary["all"] = p.aggregate_metrics([e for _, e in satisfied])
        p.write(out / "reports" / ("native_" + uuid.uuid4().hex + ".json"),
                dict(accepted=len(satisfied), panel_sha256=p.PANEL_HASH, metrics=summary))
        log("complete", accepted=len(satisfied))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("campaign", type=Path)
    run(ap.parse_args().campaign.resolve())
