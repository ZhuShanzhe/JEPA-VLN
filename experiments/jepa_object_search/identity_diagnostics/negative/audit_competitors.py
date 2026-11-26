"""Frozen active-relook matching and independent calibrated separation on multi-instance frames."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def digest(item):
    return hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()


def run(project, out):
    base = project / "jepa_reid_v21"
    sys.path[:0] = [str(project / p) for p in ("jepa_reid_v23/report", "jepa_report_v19", "jepa_reid_v21/association")]
    import appearance
    import evaluate
    import geometry_links
    import report_evidence
    cv2.setNumThreads(1)
    records = []
    trial_outputs = []
    source_hashes = {}
    selection_digests = {}
    for panel, root, detroot in [("natural", base/"data", base/"detector"),
                                 ("controlled", base/"data/controls", base/"detector/controls")]:
        preds = {r["frame_id"]:r["boxes"] for r in rows(detroot/"predictions.jsonl")}
        frames = {r["frame_id"]:r for r in rows(root/"frames.jsonl") if r["split"]=="test"}
        source_hashes[panel]={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in
                             [("frames",root/"frames.jsonl"),("detections",detroot/"predictions.jsonl")]}
        byhouse = defaultdict(list)
        for frame in frames.values():
            byhouse[frame["house"]].append(frame)
        # A trial selects an old reference and an entire new frame. The rival
        # is a scoring annotation only. Keep the nearest same-asset rival in
        # preference to other same-class instances; do not select on outcomes.
        pools = defaultdict(list)
        for house, group in sorted(byhouse.items()):
            for left, right in combinations(sorted(group,key=lambda x:(x["sequence"],x["view"])),2):
                if left["category"] != right["category"] or not right["boxes"]:
                    continue
                if panel == "controlled" and (left["sequence"] != right["sequence"] or left["stratum"]!="double"):
                    continue
                current = {b["instance"] for b in right["boxes"]}
                for old in left["boxes"]:
                    uid = old["instance"]
                    others = [b for b in right["boxes"] if b["instance"]!=uid]
                    if not others:
                        continue
                    def gap(o):
                        return float(np.linalg.norm(np.array(o["world_center"])-old["world_center"]))
                    rival=min(others,key=lambda b:(b["asset"]!=old["asset"],gap(b),b["instance"]))
                    kind="old_plus_rival" if uid in current else "rival_only_annotated"
                    near_twin=old["asset"]==rival["asset"] and gap(rival)<=1.25
                    a,b=left["frame_id"],right["frame_id"]
                    key=(panel,house,a,b,uid,rival["instance"])
                    pools[house,kind,near_twin].append((digest(["stage3-negative-v1",key]), key))
        selected=[]
        for group,pool in sorted(pools.items()):
            # The synthetic twin panel includes every eligible ordered-in-time trial.
            selected.extend(sorted(pool)[:(len(pool) if panel=="controlled" else 8)])
        selection_digests[panel]=digest(sorted(k for _,k in selected))
        @lru_cache(maxsize=32)
        def image(fid):
            bgr=cv2.imread(str(root/frames[fid]["rgb"]))
            return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        @lru_cache(maxsize=256)
        def assembled(fid,arm):
            frame=frames[fid]
            if arm=="diagnostic_gt":
                return evaluate.build(root,frame)
            return evaluate.build(root,frame,preds[fid])
        @lru_cache(maxsize=256)
        def separation(a,b,arm):
            left,_=assembled(a,arm)
            right,_=assembled(b,arm)
            return {k:set(tuple(pair) for pair in v) for k,v in geometry_links.match_views(left,right).items()}
        for kh,key in sorted(selected):
            _,h,a,b,uid,vid=key
            byid = {g["instance"]:g for g in frames[b]["boxes"]}
            old = next(g for g in frames[a]["boxes"] if g["instance"]==uid)
            rival=byid[vid]
            dist=float(np.linalg.norm(np.array(old["world_center"])-rival["world_center"]))
            has_old=uid in byid
            cat=frames[a]["category"]
            cohort="old_plus_rival" if has_old else "rival_only_annotated"
            same_asset=old["asset"]==rival["asset"]
            local = frames[a]["sequence"]==frames[b]["sequence"]
            records.append((kh,panel,h,a,b,uid,vid,cat,cohort,same_asset,dist,local))
            for arm in ("measured","diagnostic_gt"):
                (left,src_labels),(right,live_labels)=assembled(a,arm),assembled(b,arm)
                src=next((i for i,v in enumerate(src_labels) if v and v["instance"]==uid),None)
                target_indices=[i for i,v in enumerate(live_labels) if v and v["instance"]==vid]
                proofs=separation(a,b,arm)
                veto=src is not None and any((src,i) in proofs["distinct"] for i in target_indices)
                false_link=src is not None and any((src,i) in proofs["matches"] for i in target_indices)
                ref = None if src is None else appearance.build_reference(image(a), left["boxes"][src])
                outcome="withheld"
                reason="source_missing" if src is None else "source_not_storable"
                if ref is not None:
                    cv2.setRNGSeed(7)
                    cur=image(b)
                    app=appearance.match_reference(ref,cur)
                    opts=report_evidence.eligible_boxes([{"bbox":x} for x in right["boxes"]],cur.shape)
                    box,reason=report_evidence.current_support(ref,app,cur,opts)
                    if box is not None:
                        truth,=evaluate.label_boxes([np.asarray(box)*cur.shape[1]/appearance.WORK_WIDTH],frames[b]["boxes"])
                        outcome=("unverified_box" if truth is None else "old_correct" if truth["instance"]==uid
                                 else "selected_rival" if truth["instance"]==vid else "other_wrong_instance")
                trial_outputs.append(dict(trial=kh,panel=panel,house=h,category=cat,cohort=cohort,
                                          near_twin=bool(same_asset and dist<=1.25),nearby_camera_sequence=local,arm=arm,
                                          old_reference_box=src is not None,rival_box=bool(target_indices),
                                          appearance=outcome,reason=reason,distinct_proof=veto,
                                          geometric_old_rival_link=false_link))
    private=out/"private"
    private.mkdir(parents=True,exist_ok=True)
    (private/"trial_index.jsonl").write_text("".join(json.dumps(r)+"\n" for r in sorted(records)))
    (private/"trial_outcomes.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in trial_outputs))
    def summarize(rr):
        exposure=[r for r in rr if r["old_reference_box"] and r["rival_box"]]
        return dict(trials=len(rr),houses=len({r["house"] for r in rr}),source_and_rival_box=len(exposure),
                    appearance=dict(Counter(r["appearance"] for r in rr)),reasons=dict(Counter(r["reason"] for r in rr)),
                    geometry_distinct_proof=sum(r["distinct_proof"] for r in rr),
                    geometry_wrong_same_link=sum(r["geometric_old_rival_link"] for r in rr),
                    selected_rival_with_geometry_veto=sum(r["distinct_proof"] and r["appearance"]=="selected_rival" for r in rr),
                    exposed_appearance=dict(Counter(r["appearance"] for r in exposure)),
                    categories=dict(Counter(r["category"] for r in rr)))
    groups={}
    for panel in ("natural","controlled"):
        groups[panel]={}
        for arm in ("measured","diagnostic_gt"):
            pool=[r for r in trial_outputs if (r["panel"],r["arm"])==(panel,arm)]
            groups[panel][arm]={}
            for cohort in ("old_plus_rival","rival_only_annotated"):
                s=[r for r in pool if r["cohort"]==cohort]
                groups[panel][arm][cohort]={"all":summarize(s),"close_same_asset":summarize([r for r in s if r["near_twin"]]),
                                            "same_local_sequence":summarize([r for r in s if r["nearby_camera_sequence"]])}
    code_files=[Path(__file__),*[Path(mod.__file__) for mod in (appearance,evaluate,geometry_links,report_evidence)]]
    result=dict(paid_calls=0,selected_manifest_sha256=selection_digests,inputs_sha256=source_hashes,groups=groups,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files})
    (out/"results.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    for panel,p in groups.items():
        for arm,conditions in p.items():
            for c,o in conditions.items():
                vals=o["all"]
                print(panel,arm,c,vals["trials"],vals["appearance"],vals["geometry_distinct_proof"],flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("project","out"):
        parser.add_argument(key,type=Path)
    args=parser.parse_args()
    run(args.project,args.out)
