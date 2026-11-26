"""Separate historical registration, candidate supply, and a common free single-box reviewer."""
import os
from pathlib import Path
import json
import numpy as np
from PIL import Image

from diagnose_images import iou, profile, rows


def registration_box(project):
    base=project/"mapping_runtime/pilots/jepa_reid_v24_first2"
    def read(path):
        return json.loads(path.read_bytes())
    eid=read(base/"panel.json")["rows"][0]["episode_id"]
    target=Path(read(base/"comparison/accepted/second"/(eid+".json"))["result"])
    doc=read(target)
    events=[json.loads(x) for x in (target.parent/"agent/confirm_events.jsonl").read_bytes().splitlines()]
    review=next(x for x in events if x["event"]=="confirm_view_recheck" and x["reason"]=="no_current_instance_match")
    hist=doc["episodes"][0]["paper"]["agent_trace"]
    inst=next(x for x in hist["final_memory"]["instances"] if x["instance_id"]==review["instance"])
    ob,=[x for x in hist["observations"] if x["observation_id"] in inst["observation_ids"]
         and x["frame_id"]==review["frame"] and x["step"]<=review["step"]]
    return ob["bbox"]


if __name__=="__main__":
    project=Path(os.environ["NAV_ROOT"])
    base=project/"jepa_box_v26"
    private=base/"habitat/private"
    truth=rows(private/"gt.jsonl")
    history=registration_box(project)
    res={}
    for label in ("registered","before_approach","after_approach"):
        fid="clock_archived_"+label
        row=truth[fid]
        seg=np.asarray(Image.open(private/row["mask"]))
        obj=[t for t in row["boxes"] if t["pixels"]>=103]
        assert len(obj)==1
        obs={"visible_pixels":obj[0]["pixels"]}
        if label=="registered":
            obs["previous_registered_source"]=profile([{"bbox":history}],obj,seg)
        for mode in ("dino","sam","refined"):
            pool=rows(private/(mode+".jsonl"))[fid]["boxes"]
            best=max(pool,key=lambda b:iou(obj[0]["xyxy"],b["bbox"]),default=None)
            ranking=rows(base/"compare/private"/("habitat_"+mode+".jsonl"))[fid]["boxes"]
            obs[mode]={"pool_size":len(pool),"oracle_by_iou":profile([best] if best else [],obj,seg),
                       "free_top1":profile(ranking,obj,seg)}
            if label=="registered" and ranking:
                obs[mode]["top1_to_recorded_reference_iou"]=round(iou(history,ranking[0]["bbox"]),4)
        res[label]=obs
    (base/"habitat/clock_audit.json").write_text(json.dumps(res,indent=2)+"\n")
