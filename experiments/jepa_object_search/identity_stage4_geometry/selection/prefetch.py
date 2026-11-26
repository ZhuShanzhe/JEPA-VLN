"""Estimate the geometry workload from confirmation plans already issued by the agent."""
from collections import defaultdict
import json
from pathlib import Path
import sys

from replay import read, rows, sha


def main(project,out):
    prior=project/"mapping_runtime/pilots/jepa_reid_v24_first2"
    data={}
    for category,item in zip(("clock","painting"),read(prior/"panel.json")["rows"]):
        ticket=read(prior/"comparison/accepted/second"/(item["episode_id"]+".json"))
        path=Path(ticket["result"])
        assert sha(path)==ticket["result_sha256"]
        trace=rows(path.parent/"agent/action_trace.jsonl")
        event_path=path.parent/"agent/confirm_events.jsonl"
        log=rows(event_path) if event_path.exists() else []
        events=defaultdict(list)
        for r in log:
            events[r["step"]].append(r)
        arrivals={r["step"] for r in log if r["event"]=="confirm_view_recheck"}
        arrivals_planned=set()
        previous_two=set()
        starts=0
        # Use only new plans available at this step. Even if one aborts, charge
        # its requests: do not use the later unsuccessful trajectory to save work.
        for t in range(len(trace)):
            for r in events[t]:
                if r["event"]=="confirm_view_start":
                    starts+=1
                    end=t+len(r["planned_actions"])
                    if 0<end<len(trace):
                        arrivals_planned.add(end)
                        previous_two.update(range(max(t,end-2),end))
        existing={r["step"] for r in trace if r["mapping"]["is_keyframe"]}
        extra=arrivals_planned-existing
        data[category]=dict(starts=starts, endpoint_requests=len(arrivals_planned),
            endpoints_on_real_rechecks=len(arrivals_planned&arrivals),
            actual_rechecks=len(arrivals),extra_endpoint_frames=len(extra),
            preceding_cache_frames=len(previous_two),all_requested_frames=len(arrivals_planned|previous_two),
            extra_frames_including_preceding=len((arrivals_planned|previous_two)-existing),
            unused_endpoint_requests=len(arrivals_planned-arrivals))
    out.write_text(json.dumps(data,indent=2)+"\n")


if __name__=="__main__":
    main(Path(sys.argv[1]),Path(sys.argv[2]))
