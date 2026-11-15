"""Start one local training command after the frozen MOST-Nav native campaign."""
import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT = Path("/root/autodl-tmp/jepa_object_search")
CAMPAIGN = "native_20261115"
PANEL = "48f0f21020b3038121efd48e4f6dc880e47d3d1420c3a54745b3b55301d5cd3d"
LOCK = "987088778722085c9b9836df741e090783798a61a2d586cdfae58c60d11e0dc4"
EXPECTED = {"all": 300, "many_category": 150, "many_attribute": 150}
LEGACY_GPU = Path("/root/autodl-tmp/jepa_instance_bench_20261016/gpu_benchmark.lock")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(path, payload):
    tmp = path.with_suffix(".tmp." + str(os.getpid()))
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_baseline(root):
    rt = root / "mapping_runtime"
    camp = rt / "batch/campaigns" / CAMPAIGN
    vis = rt / "analysis/visibility" / CAMPAIGN
    require(digest(camp / "lock.json") == LOCK, "Frozen campaign lock differs")
    frozen = read(camp / "lock.json")
    require(frozen["panel_sha256"] == PANEL and frozen["arms"] == ["native"], "Wrong baseline")
    expected = {r["episode_id"] for r in frozen["episodes"]}
    accepted = {p.stem for p in (camp / "accepted/native").glob("*.json")}
    replayed = {p.stem for p in (vis / "verified").glob("*.json")}
    require(len(expected) == 300 and accepted <= expected and replayed <= expected,
            "Unexpected episode IDs")
    diag_path = camp / "analysis/accepted_diagnosis.json"
    diag = read(diag_path) if diag_path.is_file() else {}
    state = dict(ready=False, accepted=len(accepted), replayed=len(replayed),
                 diagnosed=diag.get("accepted", 0), expected=len(expected))
    if accepted != expected or replayed != expected:
        return {**state, "reason": "awaiting native episodes and physical replay"}
    reports = sorted((camp / "reports").glob("native_*.json"))
    ag_path = vis / "aggregate.json"
    if not reports or not diag.get("complete") or not ag_path.is_file():
        return {**state, "reason": "awaiting official aggregation and complete diagnosis"}
    ag = read(ag_path)
    if ag["verified_episodes"] != 300 or ag["accepted"] != 300:
        return {**state, "reason": "awaiting complete replay aggregate"}
    require(ag["panel_sha256"] == PANEL and ag["quarantined_replays"] == 0 and
            diag["cohort"] == "formal_accepted_native_only" and diag["expected"] == diag["accepted"] == 300
            and {k:diag["groups"][k]["n"] for k in EXPECTED} == EXPECTED, "Final diagnosis mismatch")
    for report in reports:
        official = read(report)
        require(official["accepted"] == 300 and official["panel_sha256"] == PANEL and
                set(official["metrics"]) == set(EXPECTED), "Official aggregate mismatch")
    inputs = root / "frontier_scenes/acceptance/most_many_category_attribute_frozen.jsonl"
    require(digest(inputs) == PANEL, "Frozen source manifest differs")
    replay_script = digest(rt / "analysis/visibility/visibility_replay.py")
    require(ag["replay_code_sha256"] == replay_script, "Physical replay source differs")
    for eid in expected:
        ref = read(camp / "accepted/native" / (eid + ".json"))
        replay = read(vis / "verified" / (eid + ".json"))
        result = rt / "runs" / ref["run_id"] / "result.json"
        require(Path(ref["result"]).resolve() == result.resolve() and
                ref["episode_id"] == replay["episode_id"] == eid and ref["arm"] == "native"
                and ref["namespace"] == "formal_frozen" and ref["lock"] == LOCK and
                replay["source_script_sha256"] == replay_script and
                ref["result_sha256"] == replay["source_result_sha256"] == digest(result),
                "Result/replay identity or hash mismatch")
    return {**state, "ready": True}


def gpu_available(rt, minimum_free_mib):
    def query(kind, fields):
        result = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-" + kind + "=" + fields,
                                          "--format=csv,noheader,nounits"], text=True, timeout=30)
        return [row.split(",") for row in result.splitlines() if row.strip()]

    (memory, util), = query("gpu", "memory.free,utilization.gpu")
    if int(memory) < minimum_free_mib or int(util) >= 20:
        return False, "insufficient free VRAM or GPU busy"
    mapper = os.fsencode(rt / "harness/start_mapping.py")
    for (pid,) in query("compute-apps", "pid"):
        try:
            process = Path("/proc") / str(int(pid))
            if mapper not in (process / "cmdline").read_bytes().split(b"\0"):
                return False, "another GPU job is present"
        except FileNotFoundError:
            return False, "GPU process identity not yet confirmed"
    return True, ""


def take_lock(path, stack):
    stream = stack.enter_context(path.open("a"))
    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return stream.fileno()


def update(state_dir, status, **data):
    record = dict(status=status, **data)
    path = state_dir / "status.json"
    if path.is_file():
        old = read(path)
        old.pop("updated_utc", None)
        if old == record:
            return
    record["updated_utc"] = stamp()
    write(path, record)
    print(json.dumps(record), flush=True)


def attempt(args, watch_fd):
    root, state_dir = args.project_root, args.state_dir
    rt = root / "mapping_runtime"
    state = checked_baseline(root)
    if not state["ready"]:
        update(state_dir, "waiting_baseline", **state)
        return None
    with contextlib.ExitStack() as locks:
        try:
            fds = [take_lock(rt / "eval.lock", locks)]
            if LEGACY_GPU.exists():
                fds.append(take_lock(LEGACY_GPU, locks))
        except BlockingIOError:
            update(state_dir, "waiting_gpu", reason="shared evaluation or GPU lock in use")
            return None
        ok, reason = gpu_available(rt, args.min_free_mib)
        if not ok:
            update(state_dir, "waiting_gpu", reason=reason)
            return None
        # Launch marker precedes process creation. A crash here requires reconciliation rather than a duplicate.
        write(state_dir / "started.json", dict(utc=stamp(), baseline=state, launcher_pid=os.getpid()))
        update(state_dir, "training", baseline=state)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES":"0", "HF_HUB_OFFLINE":"1",
               "TRANSFORMERS_OFFLINE":"1", "WANDB_MODE":"offline", "PYTHONUNBUFFERED":"1"}
        with (state_dir / "training.log").open("xb") as log:
            result = subprocess.run(args.command, cwd=root, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, pass_fds=(watch_fd, *fds))
        write(state_dir / "exit.json", dict(utc=stamp(), returncode=result.returncode))
        update(state_dir, "training_exited", returncode=result.returncode)
        return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--state-dir", type=Path, help="one private data-volume directory for this launch")
    parser.add_argument("--min-free-mib", type=int, default=4096)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("mode", nargs="?", choices=("check", "watch", "run"), default="check")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.project_root = args.project_root.resolve(strict=True)
    args.state_dir = (args.state_dir or args.project_root / "training_runs/first_round_gate").resolve()
    if args.mode == "check":
        print(json.dumps(checked_baseline(args.project_root)))
        if (args.state_dir / "status.json").is_file():
            print((args.state_dir / "status.json").read_text())
        return 0
    if args.command and args.command[0] == "--":
        args.command.pop(0)
    if not args.command or args.poll_seconds < 1 or args.min_free_mib < 1:
        parser.error("a local training command, positive polling interval and VRAM threshold are required")
    os.umask(0o077)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    plan = dict(project_root=str(args.project_root), command=args.command,
                min_free_mib=args.min_free_mib, campaign_lock=LOCK)
    plan_file = args.state_dir / "plan.json"
    with contextlib.ExitStack() as locks:
        try:
            watch_fd = take_lock(args.state_dir / "watch.lock", locks)
        except BlockingIOError:
            require(plan_file.exists() and read(plan_file) == plan, "Another launcher owns this directory")
            print("This training plan is already armed or running.")
            return 0
        if plan_file.exists():
            require(read(plan_file) == plan, "Existing launch plan differs")
        else:
            write(plan_file, plan)
        if (args.state_dir / "exit.json").is_file():
            return int(read(args.state_dir / "exit.json")["returncode"])
        require(not (args.state_dir / "started.json").exists(),
                "Previous training start is unaccounted for; inspect its log before any new launch")
        while True:
            try:
                code = attempt(args, watch_fd)
            except (OSError, ValueError, subprocess.SubprocessError, KeyError) as exc:
                update(args.state_dir, "needs_review", reason=str(exc))
                return 1
            if code is not None:
                return code
            if args.mode != "watch":
                return 2
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
