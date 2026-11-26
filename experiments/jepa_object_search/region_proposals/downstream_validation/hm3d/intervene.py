"""Paired source/current box interventions on identical archived Habitat RGB."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from appearance import build_reference, match_reference, WORK_WIDTH
from report_evidence import current_support, eligible_boxes, overlap


def read_rows(path):
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    assert len({r["frame_id"] for r in rows}) == len(rows)
    return {r["frame_id"]: r for r in rows}


def evaluate(ref, match, img, boxes):
    if ref is None:
        return None, "unusable_reference"
    available = eligible_boxes(boxes,img.shape)
    cv2.setRNGSeed(73)
    win, reason = current_support(ref,match,img,available)
    return None if win is None else np.asarray(win) * img.shape[1] / WORK_WIDTH, reason


def run(project, out):
    cv2.setNumThreads(1)
    prior = project/"jepa_box_v26"
    sys.path.insert(0,str(prior/"habitat"))
    from clock_audit import registration_box
    private = prior/"habitat/private"
    truth = read_rows(private/"gt.jsonl")
    manifest = read_rows(private/"manifest.jsonl")
    raw = read_rows(private/"dino.jsonl")
    ranked = read_rows(prior/"compare/private/habitat_dino.jsonl")
    images={}
    for key,f in manifest.items():
        source=private/f["rgb"]
        assert hashlib.sha256(source.read_bytes()).hexdigest()==f["rgb_sha256"]
        images[key]=np.asarray(Image.open(source).convert("RGB"))

    blind=prior/"refine_blind/private"
    labels=blind/"labels_frozen.json"
    assert hashlib.sha256(labels.read_bytes()).hexdigest()=="60ef2d9e27a7227f1a61d68c3e2ae8bbe00288fc2f7f1af1b8be81ddbb90fe6e"
    lookup=json.loads((blind/"lookup.json").read_text())
    annotated={lookup[k]:v["clock_faces"][0] for k,v in json.loads(labels.read_bytes())["rows"].items() if v.get("clock_faces")}
    clock_results={}
    for source_stage in ("registered","before_approach"):
        src="clock_archived_"+source_stage
        options={"visual_reference":annotated[src]}
        options["recorded" if source_stage=="registered" else "dino_top1"]=registration_box(project) if source_stage=="registered" else ranked[src]["boxes"][0]["bbox"]
        for mode,source_box in options.items():
            ref=build_reference(images[src],source_box)
            for live_stage in (source_stage,"after_approach"):
                dst="clock_archived_"+live_stage
                img=images[dst]
                match=match_reference(ref,img) if ref is not None else None
                for which,boxes in (("dino",raw[dst]["boxes"]),("visual_reference",[{"bbox":annotated[dst]}])):
                    winner,reason=evaluate(ref,match,img,boxes)
                    clock_results["/".join((source_stage,mode,live_stage,which))]={
                        "evidence_passed":winner is not None,"reason":reason,
                        "matches_visible_dial":bool(winner is not None and overlap(winner,annotated[dst])>=.5)}

    paint=[k for k,r in manifest.items() if r["category"]=="wall_art" and r["group"]=="target_facing"]
    positive={k:{r["instance"]:r["xyxy"] for r in truth[k]["boxes"] if r["pixels"]>=103} for k in paint}
    traces={}
    counts=Counter()
    for src in paint:
        for target,src_truth in positive[src].items():
            originals=ranked[src]["boxes"]
            src_boxes={"dino_top1":originals[0]["bbox"] if originals else None,"instance_reference":src_truth}
            for dst in paint:
                if src==dst:
                    continue
                if target in positive[dst]:
                    group="target_visible"
                elif positive[dst] and not any(b["instance"]==target for b in truth[dst]["boxes"]):
                    group="other_target_only"
                else:
                    continue
                counts[group]+=1
                img=images[dst]
                for source_arm,src_box in src_boxes.items():
                    ref=build_reference(images[src],src_box) if src_box is not None else None
                    match=match_reference(ref,img) if ref is not None else None
                    for current_arm,candidates in (("dino",raw[dst]["boxes"]),("scene_reference",[{"bbox":b} for b in positive[dst].values()])):
                        counter=traces.setdefault("/".join((group,source_arm,current_arm)),Counter())
                        winner,reason=evaluate(ref,match,img,candidates)
                        correct=winner is not None and target in positive[dst] and overlap(winner,positive[dst][target])>=.5
                        counter["trials"]+=1
                        counter["accepted"]+=int(winner is not None)
                        counter["localized_target"]+=int(correct)
                        counter["accepted_off_target"]+=int(winner is not None and not correct)
                        counter["deferred"]+=int(winner is None)

    report={"clock":clock_results,"painting_target_facing":{"distinct_views":len(paint),
            "views_meeting_minimum":sum(bool(v) for v in positive.values()),
            "eligible_directed_pairs":dict(counts),"conditions":traces},"paid_requests":0}
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("project",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    run(args.project,args.output)
