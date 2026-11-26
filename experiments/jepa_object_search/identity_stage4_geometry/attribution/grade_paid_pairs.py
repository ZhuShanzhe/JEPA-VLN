"""Join sealed geometry/identity predictions to scorer-private semantic labels."""
import hashlib
from collections import Counter
from pathlib import Path
import os,json


ROOT=Path(os.environ["NAV_ROOT"])/"identity_stage4_geometry"
BASE=ROOT/"attribution"


def read(p):
    return json.loads(p.read_bytes())


def row_key(r):
    return r["step"],r["source"]


def summarize(labels,outputs):
    pred={row_key(x):x for x in outputs}
    assert len(pred)==len(outputs)==len(labels) and set(pred)==set(map(row_key,labels))
    counts=Counter(); same_evidence=Counter(); eligible_steps=set(); claimed_ids=set(); overlap_claims=set(); reached_ids=set()
    example=[]; opportunities=[]
    for label in labels:
        r=pred[row_key(label)]
        target=set(label["targets"])
        near=set(label["official_near"]); reached_ids.update(near)
        clicks=set(label["click_ids"]); click=next(iter(clicks)) if len(clicks)==1 else None
        is_reference_task=click in target
        exact=label["reference_distance_to_observation_bbox"]
        counts["queries"]+=1
        counts["exact_unique_registered_click"]+=label["selected_distinct_clicks"]==1 and exact is not None and exact<1e-5
        counts["registered_click_on_a_task_instance"]+=is_reference_task
        counts["queries_any_task_in_report_range"]+=bool(near)
        counts["queries_with_offered_center_on_task"]+=any(x["center"] in target for x in label["live"])
        counts["queries_with_any_task_pixels_in_offered_boxes"]+=any(b["task_pixels"] for b in label["live"])
        possible=any(v["center"]==click for v in label["live"]) and is_reference_task
        offered_overlap=is_reference_task and any(str(click) in b["task_pixels"] for b in label["live"])
        counts["queries_with_box_containing_source_task_pixels_and_in_range"]+=offered_overlap and click in near
        counts["queries_with_box_containing_source_task_pixels_at_any_range"]+=offered_overlap
        counts["queries_source_task_in_official_range_regardless_of_box"]+=is_reference_task and click in near
        if offered_overlap and click in near:
            matches=[i for i,b in enumerate(label["live"]) if str(click) in b["task_pixels"]]
            opportunities.append(dict(step=label["step"],n_boxes=len(label["live"]),boxes_with_same_goal=matches,
                best_task_fraction_of_box=round(max(label["live"][i]["fraction"] for i in matches),4),
                frozen_result=r["result"],frozen_reason=r.get("reason")))
        available=possible and click in near
        counts["queries_offering_source_instance_center"]+=possible
        counts["queries_offering_source_instance_center_in_range"]+=available
        if available:
            eligible_steps.add(label["step"])
        outcome=r["result"]
        counts["frozen_"+outcome]+=1
        if outcome!="same":
            counts["source_target_in_range_under_candidate_center_without_acceptance"]+=available
            continue
        pick=r["selected_box"]
        assert type(pick)==int and 0<=pick<len(label["live"])
        box=label["live"][pick]; center=box["center"]; seen=set(map(int,box["task_pixels"]))
        confirmed=is_reference_task and center==click
        in_range=confirmed and click in near
        if in_range: claimed_ids.add(click)
        indeterminate=not is_reference_task or (not center in target and click in seen)
        same_evidence["source_click_verified_target"]+=is_reference_task
        same_evidence["selected_center_confirmed_same_task_instance"]+=confirmed
        same_evidence["confirmed_same_in_official_range"]+=in_range
        same_evidence["selected_center_on_different_task_instance"]+=is_reference_task and center in target and center!=click
        same_evidence["selected_box_contains_registered_goal_pixels_and_in_range"]+=is_reference_task and click in seen and click in near
        if is_reference_task and click in seen and click in near:
            overlap_claims.add(click)
        same_evidence["selected_box_with_zero_task_pixels"]+=not seen
        same_evidence["ambiguous_reference_or_partial_box"]+=indeterminate
        example.append(dict(step=label["step"],current_boxes=len(label["live"]),selection=pick,
            verified_source_click_on_task=is_reference_task,selected_center_same_goal=confirmed,
            selected_contains_same_goal_pixels=is_reference_task and click in seen,
            selected_box_task_fraction=round(box["fraction"],3),source_goal_in_official_range=is_reference_task and click in near,
            any_task_in_official_range=bool(near)))
    return dict(counts=counts,unique_steps_offering_clear_source_goal_in_range=len(eligible_steps),
        distinct_task_instances_in_official_range=len(reached_ids),
        distinct_in_range_registered_goals_under_selected_box_center=len(claimed_ids),
        distinct_in_range_registered_goals_with_pixels_in_selected_box=len(overlap_claims),
        positive_verdict_checks=same_evidence,accepted_queries=example,legal_visible_opportunities=opportunities)


def main():
    file=ROOT/"provider/paired_fusion_all.json"
    data=read(file)
    result=dict(prediction_file_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),purpose="Scoring of fixed paired historical queries; no report actions or navigation scores implied",arms={})
    for arm in ("keyframes_only","anchored_pnp"):
        labels=read(BASE/"private/evaluator_only"/arm/"request_labels.json")
        result["arms"][arm]=summarize(labels,[r for r in data if r["arm"]==arm])
    p=BASE/"paid_55_pair_truth.json"
    p.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    for k,v in result["arms"].items(): print(k,dict(v["counts"]),dict(v["positive_verdict_checks"]))


if __name__=="__main__":
    main()
