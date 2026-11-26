"""Locate losses inside official SAM's proposal pipeline on one fixed test RGB per house."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from PIL import Image
from proposal_coverage import matches,table
from native_regions import NativeRegions


def run(project):
    root=project/"jepa_box_v26/sam"
    frames=table(project/"jepa_reid_v21/data/frames.jsonl")
    official=table(root/"private/hssd/rows.jsonl")
    houses={}
    for key in sorted(official):
        houses.setdefault(frames[key]["house"],key)
    model=NativeRegions(project)
    from mapping.sam_backend import mask_bbox, _SOM_MIN_AREA_FRAC, _SOM_MAX_AREA_FRAC
    counts=Counter()
    for key in houses.values():
        row=frames[key]
        im=np.asarray(Image.open(project/"jepa_reid_v21/data"/row["rgb"]).convert("RGB"))
        truth=np.asarray([r["xyxy"] for r in row["boxes"]],float).reshape(-1,4)
        grid=model.grid(im)
        assert grid.shape[0]==grid.shape[1] and im.shape[0]==im.shape[1]
        raw=model.sam._generator.generate(grid)
        all_boxes=[]
        post_area=[]
        for segment in raw:
            mask=segment["segmentation"]
            area=float(np.count_nonzero(mask))/mask.size
            box={"xyxy":np.array(mask_bbox(mask))*im.shape[0]/grid.shape[0],"area":area}
            all_boxes.append(box)
            if _SOM_MIN_AREA_FRAC<=area<=_SOM_MAX_AREA_FRAC:
                post_area.append(box)
        subset=sorted(post_area,key=lambda b:b["area"],reverse=True)[:model.sam.max_masks]
        previous=[x["xyxy"] for x in official[key]["boxes"]]
        assert len(previous)==len(subset)
        np.testing.assert_allclose(np.array([b["xyxy"] for b in subset]),previous,atol=1e-6)
        counts.update(dict(frames=1,truth=len(truth),raw_boxes=len(all_boxes),area_kept_boxes=len(post_area),
                           final_boxes=len(subset),at_limit=int(len(subset)==model.sam.max_masks),
                           matches_raw=len(matches(all_boxes,truth)),matches_area=len(matches(post_area,truth)),
                           matches_filtered_capped=len(matches(subset,truth))))
    value=dict(protocol="Lexicographically first frozen HSSD test frame from each test house, regardless of whether a target is visible. Re-run the unmodified generator and validate the resulting final boxes against the original native API output before counting boxes lost. Detection thresholds inside the generator remain active at all stages. One-to-one category-instance box IoU >= 0.5.",counts=counts)
    (root/"cap_audit.json").write_text(json.dumps(value,indent=2)+"\n")
    print(json.dumps(value))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("project",type=Path)
    run(parser.parse_args().project)
