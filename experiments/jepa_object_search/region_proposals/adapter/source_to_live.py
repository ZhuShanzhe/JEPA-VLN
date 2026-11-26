"""Compare a re-generated source box and the archived source at the same real clock endpoint."""
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from appearance import build_reference,match_reference
from report_evidence import current_support,eligible_boxes


def records(path):
    return {r["frame_id"]:r for r in map(json.loads,path.read_bytes().splitlines())}


def run(project):
    base=project/"jepa_box_v26"
    private=base/"habitat/private"
    frames=records(private/"manifest.jsonl")
    sys.path.insert(0,str(base/"habitat"))
    from clock_audit import registration_box
    src="clock_archived_registered"
    end="clock_archived_after_approach"
    images={fid:np.array(Image.open(private/frames[fid]["rgb"]).convert("RGB")) for fid in (src,end)}
    result={}
    for arm in ("dino","sam","refined"):
        offered=records(base/"compare/private"/f"habitat_{arm}.jsonl")
        raw=records(private/(arm+".jsonl"))
        candidates={fid:([r for r in offered[fid]["boxes"][:3] if r["review_margin"]>0] if arm=="sam" else raw[fid]["boxes"]) for fid in images}
        choices={"archived":registration_box(project)}
        best=offered[src]["boxes"]
        if best:
            choices["free_first_region"]=best[0]["bbox"]
        result[arm]={}
        for name,box in choices.items():
            reference=build_reference(images[src],box)
            trial={"reference_built":reference is not None}
            for fid,label in ((src,"same_image_control"),(end,"fresh_near_image")):
                im=images[fid]
                live=eligible_boxes(candidates[fid],im.shape)
                if reference is None:
                    trial[label]={"supported":False,"reason":"reference_unusable"}
                else:
                    match=match_reference(reference,im)
                    winner,reason=current_support(reference,match,im,live)
                    trial[label]={"supported":winner is not None,"reason":reason,"live_boxes":len(live)}
            result[arm][name]=trial
    path=base/"adapter/source_to_live.json"
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    run(Path(os.environ["NAV_ROOT"]))
