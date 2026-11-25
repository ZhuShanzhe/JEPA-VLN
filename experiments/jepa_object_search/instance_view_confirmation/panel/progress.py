"""Read the two evaluation receipts or the last action of each unfinished attempt."""
import json
from pathlib import Path
import sys


def read(p):
    return json.loads(p.read_bytes())


def status(project):
    rt = project/"mapping_runtime"
    pilot = rt/"pilots/jepa_confirm_v18_first2"
    panel = read(pilot/"panel.json")["rows"]
    reports = []
    for row in panel:
        eid = row["episode_id"]
        approved = pilot/"comparison/accepted/second"/(eid+".json")
        replay = pilot/"replay"/(eid+".json")
        attempts = sorted((pilot/"comparison/attempts/second"/eid).glob("*/started.json"),key=lambda f:f.stat().st_mtime)
        info = dict(ordinal=row["ordinal"],status="queued",attempts=len(attempts))
        if attempts:
            recent = attempts[-1].parent
            errors = recent/"quarantined.json"
            info["status"] = "quarantined" if errors.exists() else "running"
            trace = rt/"runs"/recent.name/"agent/frontier_probe.jsonl"
            if trace.is_file():
                with trace.open("rb") as fp:
                    fp.seek(max(0,trace.stat().st_size-64000))
                    for line in reversed(fp.readlines()):
                        try:
                            info["last_step"]=json.loads(line)["step"]
                            break
                        except (KeyError, ValueError):
                            pass
        if approved.is_file():
            ep, = read(Path(read(approved)["result"]))["episodes"]
            info.update(status="independently_replayed" if replay.is_file() else "awaiting_replay",
                        official_found=ep["found_count"],quota=ep["required_target_count"],steps=ep["steps_taken"])
        reports.append(info)
    return dict(cases=reports, frozen=(pilot/"launch_receipt.json").is_file(),closed=(pilot/"closed.json").is_file())


if __name__=="__main__":
    print(json.dumps(status(Path(sys.argv[1]))))
