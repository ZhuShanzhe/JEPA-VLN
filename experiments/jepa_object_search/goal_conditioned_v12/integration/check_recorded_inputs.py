"""Check whether an archived MOST decision satisfies the V12 input contract."""
import json
from pathlib import Path
import sys

import numpy as np


def load(path):return json.loads(path.read_text())


def audit(project):
    runtime=project/"mapping_runtime"
    sources=list((runtime/"runs").glob("*/agent/jepa_selection.jsonl"))
    panel=runtime/"pilots/jepa_round10_fixed_five_20261124/comparison/accepted/second"
    accepted={Path(load(r)["result"]).resolve() for r in panel.glob("*.json")}
    service,=list((runtime/"runs").glob("jepa_v10_fixed_mapping_*"))
    totals=dict(archived_selection_logs=len(sources),decisions_logged=0,
                with_three_distinct_24_step_frontier_plans=0,
                with_verified_three_headings_before_selection=0)
    histogram={}
    for f in sorted(sources):
        records=[json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        totals["decisions_logged"]+=len(records)
        for r in records:
            plans={tuple(seq) for k,seq in r.get("supported_plans",{}).items()
                   if k.startswith("f") and k[1:].isdigit() and len(seq)==24 and set(seq)<={0,1,2}}
            if len(plans)<3:continue
            totals["with_three_distinct_24_step_frontier_plans"]+=1
            result=f.parent.parent/"result.json"
            assert result.resolve() in accepted
            ep,=load(result)["episodes"];steps=ep["paper"]["steps"];trace=[json.loads(line) for line in (f.parent/"action_trace.jsonl").read_text().splitlines()]
            t=r["step"];yaws=[];yaw=0
            for s in steps[:t+1]:
                yaws.append(yaw)
                if s["action_role"]=="navigate":
                    assert s["action"] in (1,2,3)
                    yaw+=int(s["action"]==2)-int(s["action"]==3)
            headings=set();frames=service/"mapping/diagnostics"/(ep["episode_id"]+"_frames")
            for j in range(t+1):
                if np.linalg.norm(np.subtract(steps[j]["position_before"],steps[t]["position_before"]))>.001:
                    continue
                frame=trace[j]["mapping_frame_id"];offset=(yaws[j]-yaws[t])%12
                if frame and (frames/("rgb_%06d.jpg"%frame)).is_file() and offset in (11,0,1):
                    headings.add(offset)
            histogram[len(headings)]=histogram.get(len(headings),0)+1
            totals["with_verified_three_headings_before_selection"]+=len(headings)==3
    totals["number_of_matched_headings_histogram"]=histogram
    return totals


if __name__=="__main__":
    project,dest=map(Path,sys.argv[1:])
    dest.write_text(json.dumps(audit(project),indent=2))
