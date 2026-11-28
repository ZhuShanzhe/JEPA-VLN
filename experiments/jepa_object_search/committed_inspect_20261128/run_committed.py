"""Frozen, paired Habitat approach trial. Keep the private trajectory and provider receipts on the host."""
import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRIOR = ROOT / "gpt_inspect_only_20261128"
REGIONAL = ROOT / "gpt_region_20261128"
FAMILY = ROOT / "local_target_20261128/eval_family"
sys.path[:0] = [str(PRIOR), str(REGIONAL)]
from run_inspect import physical, GPTGate
from commit_policy import CommittedInspectPolicy

PRIOR_LOCK = "0bf2ecb49036090a35fac145b4c2f1f8b927e283356583b713b756b770671863"
PRIOR_RESULTS = "f18a712effe4fa2490419c717edda180de792071d66cbbd86c632f19614fd5f4"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def journal_name(index):
    return "gpt_" + hashlib.sha256(str(index).encode()).hexdigest()[:16] + ".jsonl"


def seal(create=False):
    assert digest(PRIOR / "gate_lock.json") == PRIOR_LOCK
    assert digest(PRIOR / "accepted.json") == PRIOR_RESULTS
    locks = [FAMILY / "local_lock.json", REGIONAL / "gate_lock.json", PRIOR / "gate_lock.json"]
    for path in locks:
        for rel, value in read(path).items():
            assert digest(ROOT / rel) == value, rel
    inputs = locks + [HERE / "commit_policy.py", HERE / "run_committed.py", PRIOR / "accepted.json"]
    inputs.extend(PRIOR / "private_events" / journal_name(i) for i in range(8))
    expected = {p.relative_to(ROOT).as_posix(): digest(p) for p in inputs}
    lock = HERE / "gate_lock.json"
    if create and not lock.exists():
        with lock.open("x") as f:
            json.dump(expected, f, indent=2)
    assert read(lock) == expected, "a frozen input has changed"
    return digest(lock)


def seed_receipts(index, private):
    """GPTGate itself re-derives each full request hash and checks IDs before a cache hit."""
    filename = journal_name(index)
    previous = (PRIOR / "private_events" / filename).read_bytes()
    assert previous.endswith(b"\n")
    dst = private / filename
    if dst.exists():
        assert dst.read_bytes().startswith(previous), "paid journal prefix changed"
    else:
        tmp = dst.with_suffix(".seed.tmp")
        tmp.write_bytes(previous)
        os.replace(tmp, dst)


class TracedSim:
    def __init__(self, sim, log):
        self.sim = sim
        self.log = log
        self.positions = []

    def __getattr__(self, name):
        return getattr(self.sim, name)

    def record(self, action, collision=False):
        state = self.sim.get_agent(0).state
        pos = state.position.tolist()
        q = state.rotation
        self.log.write(json.dumps(dict(step=len(self.positions), action=action,
                                       xyz=pos, yaw=physical.policy_pose(self.sim)[2],
                                       quat_xyzw=[float(q.x),float(q.y),float(q.z),float(q.w)],
                                       collision=bool(collision))) + "\n")
        self.positions.append(pos)

    def get_sensor_observations(self):
        if not self.positions:
            self.record("start")
        return self.sim.get_sensor_observations()

    def step(self, action):
        observation = self.sim.step(action)
        self.record(action, observation.get("collided", False))
        return observation


def rollout(sim, ep, spec, channel, policy, log, trace):
    tracked = TracedSim(sim, trace)
    original_measure = physical.measure
    limit = None

    def choose(decision):
        nonlocal limit
        limit = None
        if decision is not None and decision["action_source"] == "local":
            limit = decision["commit_actions_left"]
            assert isinstance(limit, int) and limit > 0

    def pick(*args, **kwargs):
        dec, region = policy.pick(*args, **kwargs)
        choose(dec)
        return dec, region

    def fast(*args, **kwargs):
        dec = policy.fast_continue(*args, **kwargs)
        choose(dec)
        return dec

    class BoundedFollower(physical.ViewFollower):
        def next_action(self):
            nonlocal limit
            if limit == 0:
                return None, True
            act, done = super().next_action()
            if limit is not None and act is not None and not done:
                limit -= 1
            return act, done

    def verify_trace(_sim, episode, positions):
        assert _sim is tracked and tracked.positions == positions and len(positions) == 301
        return original_measure(sim, episode, positions)

    with ExitStack() as edits:
        for attr, impl in (("pick", pick), ("fast_continue", fast), ("ViewFollower", BoundedFollower),
                           ("measure", verify_trace)):
            edits.enter_context(patch.object(physical, attr, impl))
        return physical.episode_run(tracked, ep, spec, channel, log)


def run_one(i, spec, channel, private, out, version):
    with (private / f"{i}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dst = out / f"{i}_committed.json"
        if dst.exists():
            saved = read(dst)
            assert saved["frozen_sha256"] == version and saved["steps"] == 300
            return
        shard = physical.DATA / "eval/hssdov/content" / (spec["scene"] + ".json.gz")
        assert digest(shard) == spec["shard_sha256"]
        with gzip.open(shard) as f:
            ep = next(e for e in json.load(f)["episodes"] if str(e["episode_id"]) == str(spec["id"]))
        seed_receipts(i, private)
        policy = CommittedInspectPolicy(GPTGate(str(i), private))
        sim = physical.sensor_sim(ep)
        try:
            logpath, tracepath = private / f"{i}.jsonl", private / f"{i}.poses.jsonl"
            attempt = time.time_ns()
            for p in (logpath, tracepath):
                if p.exists():
                    p.rename(p.with_suffix(f".attempt_{attempt}.jsonl"))
            with logpath.open("x") as log, tracepath.open("x", buffering=1) as trace:
                result = rollout(sim, ep, spec, channel, policy, log, trace)
            assert result["steps"] == spec["budget"] == 300 and result["reachable_instances_at_start"] >= 2
            counts = {key: Counter() for key in ("gpt_mode", "gpt_evidence", "gpt_cue", "exec_status")}
            for line in logpath.read_text().splitlines():
                event = json.loads(line)
                if event.get("gpt_called"):
                    assert event["old_selected_region_id"] == event["executed_region_id"]
                    assert event["old_selected_region_key"] == event["region_key"]
                    for key in counts:
                        counts[key][str(event.get(key))] += 1
            result.update(arm="committed", frozen_sha256=version, upper_decisions=counts,
                          private_trace_sha256=digest(tracepath), private_events_sha256=digest(logpath))
            tmp = dst.with_suffix(".tmp")
            tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            os.replace(tmp, dst)
            print("complete", i, flush=True)
        finally:
            sim.close()


def main():
    p = argparse.ArgumentParser()
    op = p.add_mutually_exclusive_group(required=True)
    op.add_argument("--indices", type=int, nargs="+", choices=range(8))
    op.add_argument("--seal-only", action="store_true")
    args = p.parse_args()
    os.umask(0o077)
    version = seal(args.seal_only)
    if args.seal_only:
        print(version)
        return
    private, out = HERE / "private_events", HERE / "runs"
    private.mkdir(mode=0o700, exist_ok=True)
    out.mkdir(exist_ok=True)
    panel = read(FAMILY / "panel.json")
    channel, process = physical.start_worker(ROOT, private / f"worker_{args.indices[0]}.log")
    try:
        for i in args.indices:
            run_one(i, panel[i], channel, private, out, version)
    finally:
        channel.close()
        process.wait(timeout=60)


if __name__ == "__main__":
    main()
