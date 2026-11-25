"""Validate the real worker from the MOST interpreter against saved GPU predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from proposal_worker import ProposalWorker


def main(project):
    root=project/"jepa_reid_v21"
    frames=[json.loads(line) for line in (root/"data/frames.jsonl").read_text().splitlines()]
    rows={r["frame_id"]:r for r in map(json.loads,(root/"detector/predictions.jsonl").read_text().splitlines())}
    picked=[]
    for n in (0,1,2):
        picked.append(next(f for f in frames if f["split"]=="val" and len(rows[f["frame_id"]]["boxes"])==n))
    worker=ProposalWorker(project/"b_verification/env/bin/python",project,device="cuda")
    sizes=[]
    try:
        for frame in picked:
            image=np.asarray(Image.open(root/"data"/frame["rgb"]).convert("RGB"))
            boxes=worker.detect(image,frame["category"])
            ref=rows[frame["frame_id"]]["boxes"]
            assert len(boxes)==len(ref)
            for a,b in zip(boxes,ref):
                assert np.allclose(a["bbox"],b["xyxy"],atol=.02) and abs(a["score"]-b["score"])<.001
            sizes.append(len(boxes))
    finally:
        worker.close()
    result=dict(paid_requests=0,inputs_verified=len(sizes),proposal_counts=sizes)
    (root/"detector/worker_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("project",type=Path)
    main(p.parse_args().project.resolve())
