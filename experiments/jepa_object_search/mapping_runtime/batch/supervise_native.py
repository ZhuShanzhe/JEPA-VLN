"""Supervise one existing MOST-Nav controller; only retry evidenced transient transport faults."""
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

import paired as p


PERMANENT = re.compile(r"insufficient[_ ](?:credits|quota|funds)|payment required|"
                       r"unauthorized|authentication failed|invalid[_ ]api[_ ]key|"
                       r"all (?:keys|providers) (?:exhausted|failed)|"
                       r"context[_ ]length[_ ]exceeded|out of memory|"
                       r"no space left|HTTP(?:Error)?\s*[:=]?\s*(?:401|402|403)", re.I)
TRANSIENT = re.compile(r"connection (?:reset|aborted)|readtimeout|connecttimeout|"
                       r"remoteprotocolerror|remotedisconnected|read timed out|"
                       r"temporary failure in name resolution|temporarily unavailable|"
                       r"(?:status(?:_code)?[\s=:'\x22]+|HTTP(?:Error)?[\s:=]+)(?:502|503|504)\b|"
                       r"bad gateway|gateway timeout", re.I)


def tail(path):
    if not path.is_file():
        return ""
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 131072))
        return f.read().decode(errors="replace")


def event(out, name, **values):
    row = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), event=name, **values)
    with (out / "supervision.jsonl").open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def running(out):
    try:
        with p.mutex(out / "controller.lock"):
            return False
    except BlockingIOError:
        return True


def evidence(attempt, newest):
    reason = p.data(newest)["reason"]
    if not (reason.startswith("evaluator process exited") or
            reason in {"decision transport/model identity mismatch", "runtime degradation in episode logs",
                       "caption call unavailable, mixed episode or unexpected model",
                       "caption transport/embedding runtime failure"}):
        return False
    # Inspect only per-attempt artifacts. A prior request's transient trace cannot excuse a later fault.
    run = p.RT / "runs" / attempt.name
    raw = tail(run / "evaluator.log") + tail(attempt / "launcher.log")
    for f in [run / "agent/vlm_calls.jsonl"]:
        if f.exists():
            for call in p.lines(f):
                if call.get("error"):
                    raw += "\n" + str(call["error"])
                elif call.get("raw_response") and call["model"] != call["raw_response"].get("model"):
                    return False
    return bool(TRANSIENT.search(raw) and not PERMANENT.search(raw))


def preserve_frames(attempt, eid):
    # Mapper PNGs and its saved text/embedding memory reuse filenames on an ID retry.
    with p.mutex(p.RT / "eval.lock"):
        server, _ = p.service(18555)
        diagnostics = server / "mapping/diagnostics"
        for src, name in [(diagnostics / (eid + "_frames"), "mapper_frames_before_retry"),
                          (diagnostics / "caption_store" / eid, "caption_store_before_retry")]:
            dst = attempt / name
            if src.is_dir() and not dst.exists():
                staging = attempt / ("incomplete_archive_" + uuid.uuid4().hex)
                shutil.copytree(src, staging)
                os.rename(staging, dst)


def supervise(out):
    os.umask(0o077)
    with p.mutex(out / "supervisor.lock"):
        event(out, "watching")
        while True:
            if running(out):
                time.sleep(30)
                continue
            if (out / "PAUSE").exists():
                event(out, "manual_pause")
                return
            lock = p.data(out / "lock.json")
            row = next((r for r in lock["episodes"] if not p.accepted_path(out, "native", r["episode_id"]).exists()), None)
            if row is None:
                event(out, "all_accepted" if list((out / "reports").glob("native_*.json")) else "aggregation_needs_review")
                return
            eid = row["episode_id"]
            attempts = out / "attempts/native" / eid
            quarantines = sorted(attempts.glob("*/quarantined.json"), key=lambda f: f.stat().st_mtime_ns)
            requests = [r for r in p.lines(out / "supervision.jsonl") if r.get("event") == "retry_scheduled"]
            tried = [r for r in requests if r["episode"] == eid]
            if not quarantines or len(tried) >= 2 or any(r["attempt"] == quarantines[-1].parent.name for r in tried):
                event(out, "review_required", episode=eid, cause="no_new_quarantine_or_retry_budget")
                return
            last = quarantines[-1]
            if not evidence(last.parent, last):
                event(out, "review_required", episode=eid, cause="not_a_verified_transient")
                return
            if len(requests) >= 3 and all(r["accepted"] == requests[-1]["accepted"] for r in requests[-3:]):
                event(out, "review_required", episode=eid, cause="persistent_service_fault")
                return
            delay = [60, 300][len(tried)]
            count = len(list((out / "accepted/native").glob("*.json")))
            event(out, "retry_scheduled", episode=eid, attempt=last.parent.name, accepted=count, backoff_s=delay)
            time.sleep(delay)
            if running(out) or (out / "PAUSE").exists():
                continue
            preserve_frames(last.parent, eid)
            launch = out / ("retry_" + uuid.uuid4().hex + ".log")
            with launch.open("xb") as stream:
                result = subprocess.run([sys.executable, "-B", "-u", str(p.RT / "batch/native_campaign.py"), str(out)],
                                        stdout=stream, stderr=subprocess.STDOUT)
            event(out, "controller_exit", returncode=result.returncode)


if __name__ == "__main__":
    supervise(Path(sys.argv[1]).resolve())
