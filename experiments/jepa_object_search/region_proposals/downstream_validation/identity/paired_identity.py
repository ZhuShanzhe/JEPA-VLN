"""Paired interventions on registered and live boxes with the deployed appearance verifier."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ARMS = {"det_det":("det","det"),"gt_det":("gt","det"),"det_gt":("det","gt"),"gt_gt":("gt","gt")}


def read_rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def run(project):
    root = project/"jepa_reid_v21"
    sys.path[:0] = [str(project/p) for p in ("jepa_reid_v23/report","jepa_report_v19","jepa_reid_v21/association")]
    from appearance import WORK_WIDTH, build_reference, match_reference
    from evaluate import label_boxes
    from report_evidence import current_support, eligible_boxes

    cv2.setNumThreads(1)
    cv2.setRNGSeed(7)
    frames={row["frame_id"]:row for row in read_rows(root/"data/frames.jsonl") if row["split"]=="test"}
    detected={r["frame_id"]:r["boxes"] for r in read_rows(root/"detector/predictions.jsonl")}
    known = {}
    houses=defaultdict(list)
    for fid,row in frames.items():
        boxes = [b["xyxy"] for b in detected[fid]]
        labels = label_boxes(boxes,row["boxes"])
        known[fid] = [(i,lab["instance"]) for i,lab in enumerate(labels) if lab is not None]
        houses[row["house"]].append(row)
    # With a known instance selected for registration, take the first real
    # detector box uniquely matching it. This is deliberately favorable to
    # DINO and leaves the upper-level semantic choice outside this test.
    selected={"existing_32":[],"house_stratified":[]}
    fingerprint = []
    for h,recs in sorted(houses.items()):
        fixed={}
        pool=defaultdict(list)
        order=sorted(recs,key=lambda r:(r["sequence"],r["view"],r["frame_id"]))
        for left,right in combinations(order,2):
            if left["category"]!=right["category"] or not right["boxes"]:
                continue
            a,b=left["frame_id"],right["frame_id"]
            live_ids={v["instance"] for v in right["boxes"]}
            for idx,uid in known[a]:
                kind="same_visible" if uid in live_ids else "other_only"
                if kind not in fixed:
                    fixed[kind]=(h,a,b,uid,idx,kind)
                    stage="reference_annotated_in_live" if uid in live_ids else "other_same_category_annotated_only"
                    fingerprint.append((h,a,b,idx,stage))
            for truth in left["boxes"]:
                uid=truth["instance"]
                idx=next((i for i,label in known[a] if label==uid),None)
                kind="same_visible" if uid in live_ids else "other_only"
                key=hashlib.sha256(json.dumps(["panel-20261126",a,b,uid],sort_keys=True).encode()).hexdigest()
                pool[kind].append((key,(h,a,b,uid,idx,kind)))
        selected["existing_32"].extend(fixed.values())
        for rows in pool.values():
            selected["house_stratified"].extend(case for _,case in sorted(rows)[:12])
    checksum=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
    historical=json.loads((project/"jepa_reid_v23/verification/view_controls.json").read_bytes())
    assert checksum==historical["selection_sha256"]
    @lru_cache(maxsize=48)
    def image(fid):
        path=root/"data"/frames[fid]["rgb"]
        return cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)

    @lru_cache(maxsize=4096)
    def evaluate(a,b,uid,idx):
        truth = next(v for v in frames[a]["boxes"] if v["instance"]==uid)
        after=image(b)
        rois={"det":None if idx is None else detected[a][idx]["xyxy"],"gt":truth["xyxy"]}
        menu={"det":detected[b],"gt":frames[b]["boxes"]}
        per_pair={}
        for source,roi in rois.items():
            ref=None if roi is None else build_reference(image(a),roi)
            appearance=None if ref is None else match_reference(ref,after)
            for current,rows in menu.items():
                winner=None
                if ref is None:
                    reason="source_missing" if roi is None else "source_not_storable"
                else:
                    boxes=eligible_boxes([{"bbox":v["xyxy"]} for v in rows],after.shape)
                    cv2.setRNGSeed(7)
                    winner,reason=current_support(ref,appearance,after,boxes)
                if winner is None:
                    outcome="withheld"
                else:
                    match,=label_boxes([np.asarray(winner)*after.shape[1]/WORK_WIDTH],frames[b]["boxes"])
                    outcome="unverified" if match is None else ("correct" if match["instance"]==uid else "wrong_instance")
                per_pair[(source,current)]=(outcome,reason)
        return per_pair

    results={}
    for name,rows in selected.items():
        data=[]
        for h,a,b,uid,idx,kind in rows:
            result=evaluate(a,b,uid,idx)
            data.append((h,kind,idx is not None,{k:result[arm] for k,arm in ARMS.items()}))
        subsets={}
        for kind in ("same_visible","other_only"):
            sub=[r for r in data if r[1]==kind]
            summary={}
            for arm in ARMS:
                outcomes=Counter(r[3][arm][0] for r in sub)
                reasons=Counter(r[3][arm][1] for r in sub)
                changed=Counter((r[3]["det_det"][0],r[3][arm][0]) for r in sub)
                byhouse=defaultdict(list)
                for h,_,_,rec in sub:
                    byhouse[h].append(int(rec[arm][0]=="correct")-int(rec["det_det"][0]=="correct"))
                deltas=np.array([np.mean(x) for x in byhouse.values()])
                boot=np.random.default_rng(2611).choice(deltas,(10000,len(deltas))).mean(axis=1)
                summary[arm]={"outcomes":dict(outcomes),"reasons":dict(reasons),
                              "paired_transitions_from_det_det":{f"{a}_to_{b}":v for (a,b),v in sorted(changed.items())},
                              "house_macro_correct_rate_delta":float(deltas.mean()),
                              "house_bootstrap_95":np.quantile(boot,[.025,.975]).tolist()}
            subsets[kind]={"pairs":len(sub),"houses":len(set(r[0] for r in sub)),
                           "uniquely_matched_source_detection":sum(r[2] for r in sub),"arms":summary}
        if name == "house_stratified":
            masked=defaultdict(list)
            for h,a,b,uid,idx,kind in rows:
                if kind != "same_visible":
                    continue
                eligible_start=idx is not None
                eligible_live=any(target==uid for _,target in known[b])
                r=evaluate(a,b,uid,idx)
                masked[(eligible_start,eligible_live)].append((h,{arm:r[k][0] for arm,k in ARMS.items()}))
            condition={}
            for (start,live),items in sorted(masked.items()):
                groups=defaultdict(list)
                for h,outcomes in items:
                    groups[h].append(int(outcomes["gt_gt"]=="correct")-int(outcomes["det_det"]=="correct"))
                delta=np.array([np.mean(h) for h in groups.values()])
                condition[f"reference_{int(start)}_live_{int(live)}"]={
                    "pairs":len(items),"houses":len(groups),
                    "correct_by_arm":{arm:sum(x[arm]=="correct" for _,x in items) for arm in ARMS},
                    "house_macro_gt_minus_det_ci95":np.quantile(
                        np.random.default_rng(2611).choice(delta,(10000,len(delta))).mean(1),[.025,.975]).tolist()}
            subsets["positive_by_target_box_at_both_views"]=condition
        results[name] = subsets
    anchor=results["existing_32"]["same_visible"]["arms"]["det_det"]["outcomes"]
    assert anchor=={"correct":5,"wrong_instance":2,"withheld":9}
    assert results["existing_32"]["other_only"]["arms"]["det_det"]["outcomes"]=={"withheld":16}
    return {"heldout_houses":len(houses),"selection_sha256_existing_32":checksum,"paid_calls":0,
            "arms":"source boxes / all current category boxes; assessment uses frozen appearance verification and current support only",
            "maximum_per_house_and_condition_expanded":12,"results":results}


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("project",type=Path)
    cli.add_argument("out",type=Path)
    args=cli.parse_args()
    res=run(args.project)
    args.out.write_text(json.dumps(res,indent=2,sort_keys=True,allow_nan=False)+"\n")
    for name,data in res["results"].items():
        print(name,{k:{a:v["outcomes"] for a,v in group["arms"].items()} for k,group in data.items() if "arms" in group})
        if "positive_by_target_box_at_both_views" in data:
            print(data["positive_by_target_box_at_both_views"])
