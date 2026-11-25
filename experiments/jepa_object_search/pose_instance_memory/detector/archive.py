"""Check the same proposal supplier on archived decision RGBs without relabeling reports."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from proposals import LocalCandidateDetector, MODES, select


def main(project,device):
    sys.path.insert(0,str(project/"jepa_reid_v20/adapter"))
    from audit_archived import audit,read
    model=LocalCandidateDetector(project,device,"augment")
    pilot=project/"mapping_runtime/pilots/jepa_confirm_v18_first2"
    result=[]
    for row,prev in zip(read(pilot/"panel.json")["rows"],audit(project)["episodes"]):
        path=Path(read(pilot/"comparison/accepted/second"/(row["episode_id"]+".json"))["result"])
        category={1:"clock",14:"painting"}[row["ordinal"]]
        for check in prev["reports"]:
            step=check["step"]
            frame,=(path.parent/"agent/vlm_inputs").glob(f"*step-{step}_call-*_decision_00-current_observation.jpg")
            rgb=np.array(Image.open(frame).convert("RGB"))
            proposals=model.proposals(rgb,category,"augment")
            result.append(dict(category=category,step=step,
                               native=check["independent_boxes_on_current_rgb_at_or_before_report"],
                               counts={m:len(select(proposals,m)) for m in MODES}))
    out=project/"jepa_reid_v21/detector/archive_summary.json"
    out.write_text(json.dumps({"reports":result,"independent_identity_labels":False},indent=2)+"\n")
    print(result)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("project",type=Path)
    p.add_argument("--device",default="cuda")
    a=p.parse_args()
    main(a.project.resolve(),a.device)
