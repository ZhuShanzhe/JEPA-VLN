"""Run independently frozen detector modes on a supplemental panel."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate import summarize,atomic
from proposals import LocalCandidateDetector,MODES,select,configured


def main(project,data,out):
    rows=[json.loads(s) for s in (data/"frames.jsonl").read_text().splitlines() if s]
    mode=configured(project)
    model=LocalCandidateDetector(project,"cuda",mode)
    raw={}
    out.mkdir(exist_ok=True,parents=True)
    temp=out/"predictions.tmp"
    with temp.open("w") as f:
        for frame in rows:
            rgb=np.asarray(Image.open(data/frame["rgb"]).convert("RGB"))
            boxes=model.proposals(rgb,frame["category"],"augment")
            raw[frame["frame_id"]]=boxes
            f.write(json.dumps(dict(frame_id=frame["frame_id"],boxes=select(boxes,mode),raw_boxes=boxes))+"\n")
    assert len(rows)==len(raw)
    temp.replace(out/"predictions.jsonl")
    result=dict(selected_mode=mode,methods={m:summarize(rows,raw,m) for m in MODES})
    atomic(out/"control_detector_summary.json",result)
    print(result["methods"][mode]["totals"])


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("project","data","out"):
        p.add_argument(k,type=Path)
    a=p.parse_args()
    main(a.project.resolve(),a.data,a.out)
