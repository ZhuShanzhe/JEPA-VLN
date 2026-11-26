"""Paired reference-box and endpoint-scoring intervention for frozen HSSD view heads."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def house_interval(groups, values):
    """Paired, house-balanced mean in percentage points; one draw per house."""
    pooled = defaultdict(list)
    for group, value in zip(groups, values):
        pooled[group].append(value)
    means = np.array([np.mean(vals) for vals in pooled.values()]) * 100
    rng = np.random.default_rng(2611)
    samples = rng.choice(means, (10000, len(means))).mean(axis=1)
    return dict(houses=len(means), pooled_pp=round(float(np.mean(values))*100, 4),
                house_macro_pp=round(float(means.mean()), 4),
                house_macro_95pct_pp=np.round(np.quantile(samples, [.025, .975]), 4).tolist())


def weighted_choice(head, image, box, relative_motion):
    scores = head.score(image, box, relative_motion)
    tied = np.isclose(scores, scores.max(), atol=1e-6)
    return tied / np.count_nonzero(tied)


def run(project):
    source = project / "jepa_reid_v21"
    active = project / "jepa_reid_v25/active"
    sys.path[:0] = [str(active), str(source/"model")]
    from active_view_score import ActiveViewScorer
    from train_views import displacement, rows
    from encode_proposals import assignments
    weights = {name: active/"weights"/(name+".npz") for name in ("jepa", "direct")}
    heads = {name: ActiveViewScorer(file) for name,file in weights.items()}
    manifest = rows(source/"data/frames.jsonl")
    test = [r for r in manifest if r["split"] == "test"]
    assert len(test) == 288 and len({r["house"] for r in test}) == 16
    assert not ({r["house"] for r in test} & {r["house"] for r in manifest if r["split"] != "test"})
    predictions = {r["frame_id"]:r for r in rows(source/"detector/predictions.jsonl")}
    sequences = defaultdict(list)
    for r in test:
        gt = [k["xyxy"] for k in r["boxes"]]
        det = [k["xyxy"] for k in predictions[r["frame_id"]]["boxes"]]
        r["det"] = {r["boxes"][gi]["instance"]: det[di] for di,gi in assignments(det,gt).items()}
        r["gt"] = {k["instance"]:k["xyxy"] for k in r["boxes"]}
        assert len(gt) == len(r["gt"])
        r["camera_xyz"] = np.array(r["T_world_from_camera"])[:3,3]
        sequences[r["sequence"]].append(r)
    cases=[]
    for seq in sequences.values():
        assert len(seq) == 3
        for r in seq:
            alts=[v for v in seq if v is not r]
            motion = [displacement(r,v) for v in alts]
            img = np.array(Image.open(source/"data"/r["rgb"]).convert("RGB"))
            for obj, box in r["gt"].items():
                end={key:np.array([obj in v[key] for v in alts]) for key in ("gt","det")}
                assert (end["gt"]>=end["det"]).all()
                c = dict(house=r["house"], matched=obj in r["det"], decisions={}, outcomes={}, diff_ends={})
                c["small"] = (box[2]-box[0])*(box[3]-box[1])/(img.shape[0]*img.shape[1]) <= .02
                for end_mode,y in end.items():
                    c["diff_ends"][end_mode] = y[0] != y[1]
                for region in (["gt","det"] if c["matched"] else ["gt"]):
                    for model,head in heads.items():
                        votes = weighted_choice(head,img,r[region][obj],motion)
                        c["decisions"][region,model] = votes
                        for end_mode,y in end.items():
                            c["outcomes"][region,model,end_mode] = float(votes @ y)
                cases.append(c)
    def summarize(cohort):
        region_options = ["det","gt"] if all(c["matched"] for c in cohort) else ["gt"]
        house=[c["house"] for c in cohort]
        output=dict(cases=len(cohort),houses=len(set(house)), metrics={}, contrasts={})
        for end_mode in ("det","gt"):
            output["different_endpoint_"+end_mode] = sum(bool(c["diff_ends"][end_mode]) for c in cohort)
        for region in region_options:
            for model in heads:
                for end_mode in ("det","gt"):
                    a=[c["outcomes"][region,model,end_mode] for c in cohort]
                    output["metrics"][region+"_"+model+"_"+end_mode] = dict(hits=sum(a),rate=np.mean(a))
                gaps = [c["outcomes"][region,model,"gt"] - c["outcomes"][region,model,"det"] for c in cohort]
                output["contrasts"][region+"_"+model+"_endpoint_gt_minus_detected"] = house_interval(house,gaps)
            for end_mode in ("det","gt"):
                gap = [c["outcomes"][region,"jepa",end_mode]-c["outcomes"][region,"direct",end_mode] for c in cohort]
                output["contrasts"][region+"_"+end_mode+"_jepa_minus_direct"] = house_interval(house,gap)
        if len(region_options)==2:
            for model in heads:
                output["decision_changed_"+model] = sum(not np.array_equal(c["decisions"]["gt",model], c["decisions"]["det",model]) for c in cohort)
                for end_mode in ("gt","det"):
                    diffs=[c["outcomes"]["gt",model,end_mode]-c["outcomes"]["det",model,end_mode] for c in cohort]
                    output["contrasts"][model+"_"+end_mode+"_gt_source_minus_det_source"] = house_interval(house,diffs)
            for end_mode in ("det","gt"):
                did=[(c["outcomes"]["gt","jepa",end_mode]-c["outcomes"]["gt","direct",end_mode])-
                     (c["outcomes"]["det","jepa",end_mode]-c["outcomes"]["det","direct",end_mode]) for c in cohort]
                output["contrasts"][end_mode+"_method_delta_change_when_replacing_source"] = house_interval(house,did)
        return output

    matched=[c for c in cases if c["matched"]]
    missed=[c for c in cases if not c["matched"]]
    assert (len(cases),len(matched),len(missed)) == (661,338,323)
    out=dict(protocol="Reference appears in V21 annotated HSSD test group; all two alternative endpoints and motions fixed. gt reference is privileged. det reference is one-to-one class detector box matched to annotated instance at IoU >= 0.5. gt endpoint is presence of qualifying annotation; det endpoint is detection assigned to that same instance. Ties averaged as in frozen V25.",
             paid_calls=0, frames=len(test),sequences=len(sequences),
             checkpoints_sha256={k: hashlib.sha256(v.read_bytes()).hexdigest() for k,v in weights.items()},
             matched_source=summarize(matched), matched_small_gt=summarize([c for c in matched if c["small"]]),
             matched_one_endpoint_annotated=summarize([c for c in matched if c["diff_ends"]["gt"]]),
             matched_one_endpoint_detected=summarize([c for c in matched if c["diff_ends"]["det"]]),
             gt_only_source=summarize(missed), all_gt_sources=summarize(cases))
    original=json.loads((active/"detector_transfer.json").read_text())
    for name in heads:
        assert np.isclose(out["matched_source"]["metrics"]["det_"+name+"_det"]["rate"],
                          original["all"]["metrics"][name]["prob_same_instance_detected"])
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("project",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    result=run(args.project)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result["matched_source"]["metrics"],indent=2), flush=True)
