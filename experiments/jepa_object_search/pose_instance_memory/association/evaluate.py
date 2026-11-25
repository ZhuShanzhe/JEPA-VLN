"""Compare geometric identity constraints on actual detections and on labeled boxes."""
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path
import argparse
import json
import numpy as np

from geometry_links import match_views, overlap, surface_point


def read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l]


def label_boxes(boxes, gt):
    # Independent assignment: multiple fragments mapped to one GT can expose a
    # false "distinct" claim. Unmatched detections remain an explicit unknown.
    labels = []
    for box in boxes:
        hits = [g for g in gt if overlap(np.array(box),np.array(g["xyxy"])) >= .5]
        labels.append(hits[0] if len(hits)==1 else None)
    return labels


def build(root, fr, det=None):
    gt=fr["boxes"]
    boxes = [b["xyxy"] for b in (gt if det is None else det)]
    depth=np.load(root/fr["depth"],allow_pickle=False)
    K,pose=np.array(fr["K"]),np.array(fr["T_world_from_camera"])
    points=[surface_point(depth,box,K,pose) for box in boxes]
    return dict(frame_id=fr["frame_id"], boxes=boxes, c2w=pose,K=K,points=points), (gt if det is None else label_boxes(boxes,gt))


def tally(c, tag, a, b, want_same):
    c[tag+"_all"] += 1
    if a is None or b is None:
        c[tag+"_unknown"]+=1
    elif (a["instance"]==b["instance"])==want_same:
        c[tag+"_correct"]+=1
    else:
        c[tag+"_wrong"]+=1


def evaluate(root, pred, data, mode):
    c=Counter()
    for seq in data:
        frames=[build(root,r,None if pred is None else pred[r["frame_id"]]["boxes"]) for r in seq]
        for fr,(view,gt) in zip(seq,frames):
            distinct=match_views(view,view)["same_frame_distinct_left"]
            c["frames"]+=1
            c["frames_with_multiple_ground_truth"]+=len(fr["boxes"])>=2
            for i,j in distinct:
                tally(c,"co_visible_distinct",gt[i],gt[j],False)
            recovered={tuple(sorted((gt[i]["instance"],gt[j]["instance"]))) for i,j in distinct
                       if gt[i] and gt[j] and gt[i]["instance"]!=gt[j]["instance"]}
            c["all_gt_co_visible_pairs"]+=len(fr["boxes"])*(len(fr["boxes"])-1)//2
            c["distinct_gt_pairs_recovered_in_frame"]+=len(recovered)
        for (v1,(f1,g1)),(v2,(f2,g2)) in combinations(zip(seq,frames),2):
            actual_links=match_views(f1,f2)
            for i,(box,pt) in enumerate(zip(f1["boxes"],f1["points"])):
                if not len(f2["boxes"]):
                    continue
                ious=np.array([overlap(np.array(box),np.array(other)) for other in f2["boxes"]])
                j=int(ious.argmax())
                if ious[j]>=.5:
                    tally(c,"baseline_pixel_same",g1[i],g2[j],True)
                if pt is not None:
                    distance=[np.linalg.norm(np.asarray(pt)-other) if other is not None else np.inf for other in f2["points"]]
                    j=int(np.argmin(distance))
                    # 0.3473 m position-only threshold from the already frozen V20 gallery.
                    if distance[j]<=.347302806383428:
                        tally(c,"baseline_3d_same",g1[i],g2[j],True)
            for key,tag,want in (("matches","cross_view_same",True),("distinct","cross_view_distinct",False)):
                for i,j in actual_links[key]:
                    tally(c,tag,g1[i],g2[j],want)
            expected_same={(a["instance"],b["instance"]) for a in v1["boxes"] for b in v2["boxes"]
                           if a["instance"]==b["instance"]}
            recovered_same={(g1[i]["instance"],g2[j]["instance"]) for i,j in actual_links["matches"]
                            if g1[i] and g2[j] and g1[i]["instance"]==g2[j]["instance"]}
            c["gt_same_opportunities"]+=len(expected_same)
            c["gt_same_recovered"]+=len(recovered_same)
            recovered_diff={(g1[i]["instance"],g2[j]["instance"]) for i,j in actual_links["distinct"]
                            if g1[i] and g2[j] and g1[i]["instance"]!=g2[j]["instance"]}
            paired={k[0] for k in expected_same}
            c["persistent_distinct_opportunities"]+=len(paired)*(len(paired)-1)
            c["persistent_distinct_recovered"]+=sum(a in paired and b in paired for a,b in recovered_diff)
            src={a["instance"]:a for a in v1["boxes"]}
            for a,b in combinations(sorted(paired),2):
                if src[a]["asset"]==src[b]["asset"] and np.linalg.norm(np.asarray(src[a]["world_center"])-src[b]["world_center"])<=1.25:
                    c["close_same_asset_distinct_opportunities"]+=2
                    c["close_same_asset_distinct_recovered"]+=int((a,b) in recovered_diff)+int((b,a) in recovered_diff)
    return dict(c)


def main(root, pred_file, output):
    rows = [r for r in read(root/"frames.jsonl") if r["split"]=="test"]
    pred = {r["frame_id"]:r for r in read(pred_file)}
    sequences=defaultdict(list)
    for row in rows:
        assert row["frame_id"] in pred
        sequences[row["sequence"]].append(row)
    seq=[sorted(s,key=lambda r:r["view"]) for s in sequences.values()]
    assert all(len(s)==3 for s in seq)
    result=dict(test_houses=len({r["house"] for r in rows}),sequences=len(seq),protocol="ground truth only for scoring; independently IoU>=0.50 uniquely labeled predictions; directional cross-frame pairs, three view pairs per sequence",methods={})
    for kind,value in (("measured_boxes",pred),("oracle_boxes_diagnostic",None)):
        result["methods"][kind]=evaluate(root,value,seq,kind)
    if all("layout" in s[0] for s in seq):
        result["controlled_layouts"]={}
        for cat,condition in sorted({(s[0]["category"],s[0]["stratum"]) for s in seq}):
            subset=[s for s in seq if s[0]["category"]==cat and s[0]["stratum"]==condition]
            result["controlled_layouts"][cat+"_"+condition]=evaluate(root,pred,subset,"measured_boxes")
    by_house=defaultdict(list)
    for s in seq:
        by_house[s[0]["house"]].append(s)
    strata=[evaluate(root,pred,v,"measured_boxes") for v in by_house.values()]
    ratio=lambda k,n:[c[k]/c[n] for c in strata if c.get(n,0)>0 and k in c]
    for k,n in (("gt_same_recovered","gt_same_opportunities"),
                ("persistent_distinct_recovered","persistent_distinct_opportunities")):
        values=ratio(k,n)
        result.setdefault("house_weighted_coverage",{})[k]=dict(mean=float(np.mean(values)) if values else None,
                                                               houses=len(values))
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("data","predictions","out"):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    main(a.data,a.predictions,a.out)
