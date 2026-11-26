"""Replay single-reference and causal two-reference evidence under identical current images."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


def run(project, root, out):
    sys.path[:0] = [str(project / n) for n in ("jepa_reid_v23/report", "jepa_reid_v21/association",
                                               "jepa_reid_v21/detector","jepa_report_v19")]
    from appearance import WORK_WIDTH, build_reference, match_reference
    from report_evidence import current_support, eligible_boxes
    from evaluate import label_boxes
    from proposal_worker import ProposalWorker
    cv2.setNumThreads(1)
    frames = [json.loads(line) for line in (root/"frames.jsonl").read_text().splitlines()]
    grouped = defaultdict(dict)
    for row in frames:
        grouped[row["group"]][row["stage"]] = row
    @lru_cache(maxsize=8)
    def rgb(name):
        with Image.open(root/"observations"/(name+".png")) as im:
            return np.array(im.convert("RGB"))
    pred_path = root / "dino.jsonl"
    if not pred_path.exists():
        worker = ProposalWorker(project/"b_verification/env/bin/python", project, device="cuda")
        try:
            with pred_path.with_suffix(".tmp").open("w") as handle:
                for row in frames:
                    boxes = worker.detect(rgb(row["name"]), row["category"])
                    handle.write(json.dumps(dict(name=row["name"], boxes=[b["bbox"] for b in boxes]))+"\n")
        finally:
            worker.close()
        pred_path.with_suffix(".tmp").replace(pred_path)
    pred = {row["name"]:row["boxes"] for row in map(json.loads, pred_path.read_text().splitlines())}
    assert set(pred) == {r["name"] for r in frames}
    def menu(row, mode):
        return pred[row["name"]] if mode == "dino" else [b["xyxy"] for b in row["boxes"]]
    def seed_ref(row, mode):
        if row is None:
            return None
        rois = menu(row, mode)
        labels = label_boxes(rois, row["boxes"])
        for roi, label in zip(rois, labels):
            if label is not None and label["instance"] == row["uid"]:
                return build_reference(rgb(row["name"]), roi)
        return None
    def find(ref, row, mode):
        if ref is None:
            return None
        image = rgb(row["name"])
        rois = eligible_boxes([{"bbox":b} for b in menu(row, mode)], image.shape)
        cv2.setRNGSeed(7)
        box, _ = current_support(ref, match_reference(ref,image), image, rois)
        return box
    def decode(row, box):
        if box is None:
            return "withheld"
        scale = rgb(row["name"]).shape[1] / WORK_WIDTH
        label, = label_boxes([np.array(box)*scale], row["boxes"])
        if label is None:
            return "unverified"
        return "correct" if label["instance"] == row["uid"] else "other_instance"
    def combine(first, second):
        if first is not None and second is not None and first != second:
            return None
        return first if first is not None else second

    raw = []
    bridges = defaultdict(Counter)
    for gid, views in sorted(grouped.items()):
        source, bridge = views["anchor"], views.get("bridge")
        for mode in ("dino", "label_box"):
            ref = seed_ref(source,mode)
            bridge_ref = None
            if bridge is not None and ref is not None:
                found = find(ref,bridge,mode)
                bridges[mode][decode(bridge,found)] += 1
                if found is not None:
                    im = rgb(bridge["name"])
                    # No annotation check here: a wrong accepted bridge may contaminate memory.
                    bridge_ref = build_reference(im, np.array(found)*im.shape[1]/WORK_WIDTH)
            else:
                bridges[mode]["no_anchor_reference" if ref is None else "no_bridge_pose"] += 1
            ideal_bridge = seed_ref(bridge, mode) if ref is not None else None
            for stage in ("same_side", "opposite_side", "closer"):
                if stage not in views:
                    continue
                row = views[stage]
                a, b, c = (find(item,row,mode) for item in (ref,bridge_ref,ideal_bridge))
                outcomes = {k:decode(row,box) for k,box in (("first",a),("causal",combine(a,b)),
                                                             ("known_bridge",combine(a,c)))}
                raw.append(dict(mode=mode,group=gid,house=row["house"],stage=stage,category=row["category"],
                                visible=any(r["instance"]==row["uid"] for r in row["boxes"]),
                                anchor_stored=ref is not None,bridge_stored=bridge_ref is not None,
                                movement=row["movement_m"],outcomes=outcomes))
    (root/"outcomes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in raw))
    def summarize(cases):
        return dict(n=len(cases),outcomes={arm:dict(Counter(r["outcomes"][arm] for r in cases))
                               for arm in ("first","causal","known_bridge")})
    total = {m:{} for m in bridges}
    for mode, dest in total.items():
        items = [r for r in raw if r["mode"]==mode]
        yes = [r for r in items if r["visible"]]
        dest["target_annotated"] = summarize(yes)
        dest["target_below_visibility_rule_or_absent"] = summarize([r for r in items if not r["visible"]])
        dest["registered_and_visible"] = summarize([r for r in yes if r["anchor_stored"]])
        dest["by_goal_visible"] = {k:summarize([r for r in yes if r["category"]==k]) for k in ("clock","wall_art")}
        dest["by_view_visible"] = {k:summarize([r for r in yes if r["stage"]==k]) for k in ("same_side","opposite_side","closer")}
        dest["bridge_outcomes"] = dict(bridges[mode])
        dest["causal_transition_visible"] = dict(Counter(r["outcomes"]["first"]+"_to_"+r["outcomes"]["causal"] for r in yes))
        grouped_delta = defaultdict(list)
        for r in yes:
            grouped_delta[r["house"]].append(int(r["outcomes"]["causal"]=="correct")-int(r["outcomes"]["first"]=="correct"))
        delta = [np.mean(v) for v in grouped_delta.values()]
        if delta:
            draws = np.random.default_rng(719).choice(delta,size=(10000,len(delta))).mean(axis=1)
            dest["paired_house_macro_delta"] = dict(n_houses=len(delta),mean=float(np.mean(delta)),ci95=np.quantile(draws,[.025,.975]).tolist())
    result = dict(houses=len({r["house"] for r in frames}),physical_instances=len(grouped),
                  images=len(frames), stage_counts=dict(Counter(r["stage"] for r in frames)),
                  instances_by_category=dict(Counter(views["anchor"]["category"] for views in grouped.values())),
                  unavailable_views=len(json.loads((root/"skipped.json").read_text())),
                  pose_protocol_sha256=json.loads((root/"protocol.json").read_text())["selection_sha256"],
                  private_manifest_sha256=hashlib.sha256((root/"frames.jsonl").read_bytes()).hexdigest(),
                  paid_calls=0, results=total)
    assert len(raw) == 2 * sum(result["stage_counts"].get(x,0) for x in ("same_side","opposite_side","closer"))
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("revisit scoring completed", result["images"], result["physical_instances"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for key in ("project","data","out"):
        ap.add_argument(key,type=Path)
    args = ap.parse_args()
    run(args.project.resolve(),args.data.resolve(),args.out.resolve())
