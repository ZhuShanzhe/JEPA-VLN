"""Check selective live-image support on a small fixed sample of held-out HSSD houses."""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


def run(project):
    root = project / "jepa_reid_v21"
    sys.path[:0] = [str(project/p) for p in ("jepa_reid_v23/report","jepa_report_v19","jepa_reid_v21/association")]
    from appearance import WORK_WIDTH, build_reference, match_reference
    from evaluate import label_boxes
    from report_evidence import current_support, eligible_boxes
    frames = [json.loads(l) for l in (root/"data/frames.jsonl").read_text().splitlines()]
    detected = {r["frame_id"]:r["boxes"] for l in (root/"detector/predictions.jsonl").read_text().splitlines()
                if (r:=json.loads(l))}
    houses = defaultdict(list)
    for r in frames:
        if r["split"] == "test":
            houses[r["house"]].append(r)
    outcome = defaultdict(Counter)
    chosen = []
    for house, records in sorted(houses.items()):
        fixed = set()
        ordered = sorted(records,key=lambda r:(r["sequence"],r["view"],r["frame_id"]))
        for left,right in combinations(ordered,2):
            if left["category"] != right["category"] or not right["boxes"]:
                continue
            # Select from ordinary detections; score each physical instance only
            # against a reference from the same house and category.
            left_boxes = [b["xyxy"] for b in detected[left["frame_id"]]]
            truth = {a["instance"] for a in right["boxes"]}
            for i, lab in enumerate(label_boxes(left_boxes,left["boxes"])):
                if lab is None:
                    continue
                uid = lab["instance"]
                kind = "reference_annotated_in_live" if uid in truth else "other_same_category_annotated_only"
                if kind in fixed:
                    continue
                fixed.add(kind)
                chosen.append((house,left["frame_id"],right["frame_id"],i,kind))
                record = outcome[kind]
                record["pairs"] += 1
                before = (root/"data"/left["rgb"]).read_bytes()
                after = cv2.cvtColor(cv2.imread(str(root/"data"/right["rgb"])),cv2.COLOR_BGR2RGB)
                ref = build_reference(before,left_boxes[i])
                if ref is None:
                    record["reference_not_storable"] += 1
                    continue
                old = match_reference(ref,after)
                boxes = eligible_boxes([{"bbox":b["xyxy"]} for b in detected[right["frame_id"]]],after.shape)
                new, _ = current_support(ref,old,after,boxes)
                for name,box in (("old",old.box if old.matched else None),("new",new)):
                    if box is None:
                        record[name+"_withheld"] += 1
                        continue
                    mapped, = label_boxes([np.asarray(box)*after.shape[1]/WORK_WIDTH],right["boxes"])
                    result = "unverified_box" if mapped is None else ("correct_instance" if mapped["instance"]==uid else "other_instance")
                    record[name+"_"+result] += 1
            if len(fixed)==2:
                break
    sha = hashlib.sha256(json.dumps(chosen,sort_keys=True).encode()).hexdigest()
    return dict(heldout_houses=len(houses),pairs=len(chosen),selection_sha256=sha,
                selected_before_matching=True,paid_calls=0,
                counts={k:dict(v) for k,v in outcome.items()})


if __name__ == "__main__":
    result = run(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
