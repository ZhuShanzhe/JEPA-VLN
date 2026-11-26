"""Test original-vs-copy reference crops on exactly the same live original-only image."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def read(p):
    return [json.loads(line) for line in p.read_bytes().splitlines() if line]


def run(project, out):
    sys.path[:0] = [str(project/p) for p in ("jepa_reid_v23/report","jepa_report_v19","jepa_reid_v21/association")]
    import appearance
    import evaluate
    import report_evidence
    cv2.setNumThreads(1)
    root=project/"jepa_reid_v21/data/controls"
    det_file=project/"jepa_reid_v21/detector/controls/predictions.jsonl"
    frames={x["frame_id"]:x for x in read(root/"frames.jsonl")}
    detector={x["frame_id"]:x["boxes"] for x in read(det_file)}
    by_layout=defaultdict(list)
    for f in frames.values():
        assert f["split"]=="test"
        by_layout[f["layout"]].append(f)
    @lru_cache(maxsize=32)
    def image(fid):
        return cv2.cvtColor(cv2.imread(str(root/frames[fid]["rgb"])),cv2.COLOR_BGR2RGB)
    @lru_cache(maxsize=256)
    def ref(fid, arm, uid):
        src=frames[fid]
        boxes=src["boxes"] if arm=="diagnostic_gt" else detector[fid]
        labels=boxes if arm=="diagnostic_gt" else evaluate.label_boxes([b["xyxy"] for b in boxes],src["boxes"])
        for box,label in zip(boxes,labels):
            if label is not None and label["instance"]==uid:
                return appearance.build_reference(image(fid),box["xyxy"])
        return None
    def verify(reference,cur,arm):
        if reference is None:
            return "withheld","source_missing_or_not_storable"
        fid=cur["frame_id"]
        rgb=image(fid)
        boxes=cur["boxes"] if arm=="diagnostic_gt" else detector[fid]
        menu=report_evidence.eligible_boxes([{"bbox":b["xyxy"]} for b in boxes],rgb.shape)
        cv2.setRNGSeed(7)
        app=appearance.match_reference(reference,rgb)
        winner,reason=report_evidence.current_support(reference,app,rgb,menu)
        if winner is None:
            return "withheld", reason
        hit,=evaluate.label_boxes([np.asarray(winner)*rgb.shape[1]/appearance.WORK_WIDTH],cur["boxes"])
        return ("original_box" if hit else "unverified"),reason
    records=[]
    excluded=Counter()
    for layout,ff in sorted(by_layout.items()):
        for src in sorted((f for f in ff if f["stratum"]=="double"),key=lambda f:f["view"]):
            for live in sorted((f for f in ff if f["stratum"]=="single"),key=lambda f:f["view"]):
                if len(src["boxes"])!=2 or len(live["boxes"])!=1:
                    excluded["missing_unique_pair"]+=1
                    continue
                original,=live["boxes"]
                if "synthetic" in original["instance"] or original["instance"] not in {o["instance"] for o in src["boxes"]}:
                    excluded["inconsistent_source"]+=1
                    continue
                twin,=[o for o in src["boxes"] if o["instance"]!=original["instance"]]
                assert "synthetic" in twin["instance"] and twin["asset"]==original["asset"]
                for arm in ("measured","diagnostic_gt"):
                    outputs=[]
                    for obj in (original,twin):
                        reference=ref(src["frame_id"],arm,obj["instance"])
                        outputs.append(verify(reference,live,arm))
                    records.append(dict(layout=layout,house=src["house"],source=src["frame_id"],live=live["frame_id"],
                                        same_pose=src["view"]==live["view"],category=src["category"],arm=arm,
                                        original=outputs[0][0],twin=outputs[1][0],
                                        original_reason=outputs[0][1],twin_reason=outputs[1][1]))
    private=out/"private"
    private.mkdir(exist_ok=True,parents=True)
    (private/"paired_lookalikes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    groups={}
    for arm in ("measured","diagnostic_gt"):
        groups[arm]={}
        for name, pred in [("all",lambda r:True),("same_camera",lambda r:r["same_pose"]),("changed_camera",lambda r:not r["same_pose"]),
                           ("clock",lambda r:r["category"]=="clock"),("wall_art",lambda r:r["category"]=="wall_art")]:
            ss=[r for r in records if r["arm"]==arm and pred(r)]
            groups[arm][name]=dict(trials=len(ss),houses=len({r["house"] for r in ss}),layouts=len({r["layout"] for r in ss}),
                                   original=dict(Counter(r["original"] for r in ss)),twin=dict(Counter(r["twin"] for r in ss)),
                                   joint=dict(Counter("/".join((r["original"],r["twin"])) for r in ss)))
    print(json.dumps(groups["measured"]["all"]),flush=True)
    print(json.dumps(groups["diagnostic_gt"]["all"]),flush=True)
    summary=dict(paid_calls=0,groups=groups,excluded=dict(excluded),
                 frozen_inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/"frames.jsonl",det_file,Path(__file__))},
                 eligible_trials_sha256=hashlib.sha256(json.dumps(sorted((r["layout"],r["source"],r["live"]) for r in records if r["arm"]=="measured")).encode()).hexdigest())
    (out/"lookalike_results.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("project","out"):
        p.add_argument(k,type=Path)
    args=p.parse_args()
    run(args.project,args.out)
