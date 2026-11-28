"""Diagnose the frozen confirmation head on measured frames, separately from view acquisition."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "refined_confirmation_20261129"
OLD = HERE.parent / "reactive_view_20261129"
sys.path.insert(0, str(RUN))
from clarity_refit import ClarityEstimator, crop_features
from frame_confirm import FrameConfirmer, thresholds
from score_paired import box_grade


def distribution(values):
    if not values:
        return {}
    return dict(zip(("min", "q25", "median", "q75", "max"), [round(float(v), 3) for v in np.percentile(values, [0,25,50,75,100])]))


def mask_measure(mask, depth):
    y, x = np.nonzero(mask)
    if not len(x):
        return {}
    rect = [int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1]
    l,t,r,b=rect
    z=depth[y,x]
    return dict(true_pixels=len(x), z_m=float(np.median(z)),
                envelope_fill=len(x)/((r-l)*(b-t)), horizontal_span=r-l, vertical_span=b-t,
                mask_touches_image_edge=bool(l==0 or t==0 or r==mask.shape[1] or b==mask.shape[0]))


def eligible(mask, box):
    if box is None:
        return False
    n,overlap,iou=box_grade(mask,box)
    return n>=369 and overlap>=.5*n and iou>=.30


def source_support(mask, current_depth, pose, source_depth, source_mask, ref):
    """Backproject currently visible labeled pixels to the original observation for offline diagnosis."""
    y,x=np.nonzero(mask)
    z=current_depth[y,x]
    right=(x+.5-96)*z/96
    up=(96-y-.5)*z/96
    dx,dz,a=pose
    u3=dx+right*np.cos(a)-z*np.sin(a)
    z3=dz+right*np.sin(a)+z*np.cos(a)
    u=np.floor(96+96*u3/np.maximum(.001,z3)).astype(int)
    v=np.floor(96-96*up/np.maximum(.001,z3)).astype(int)
    l,t,r,b=ref.sourcebox
    total=(b-t)*(r-l)
    stride=max(1,int(np.ceil(total/6000)))
    if stride!=1:
        # In this diagnostic, do not invent foreground weights between sampled points.
        weight=np.full(total,np.nan)
    else:
        weight=np.zeros(total)
    weight[::stride]=ref.w
    grid=np.zeros_like(source_depth)
    grid[t:b,l:r]=weight.reshape(b-t,r-l)
    geometry=np.zeros(len(x),bool)
    weighted=geometry.copy()
    for du in (-1,0,1):
        for dv in (-1,0,1):
            xx,yy=u+du,v+dv
            inside=(z3>.05)&(xx>=0)&(xx<192)&(yy>=0)&(yy<192)
            jj=np.flatnonzero(inside); i,k=yy[jj],xx[jj]
            match=source_mask[i,k].astype(bool)&(np.abs(source_depth[i,k]-z3[jj])<=.07+.045*z3[jj])
            geometry[jj]|=match
            weighted[jj]|=match&(grid[i,k]>=.35)
    return {"fraction_linked_to_any_compatible_initial_target_surface":float(geometry.mean()),
            "fraction_linked_to_compatible_retained_initial_target_surface":float(weighted.mean())}


def frozen_scores(rgb,depth,pose,mask,event,ref,model,initial_depth,initial_mask):
    box=event["box"]
    estimate=ref.observe(rgb,depth,pose)
    assert list(estimate["box"])==list(box)
    predicted=model.predict(rgb,depth,box,ref,pose,estimate)
    assert abs(predicted-event["p_clear"])<1e-6
    cutoff=thresholds()["threshold"]
    y,x=np.nonzero(mask)
    true_box=[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]
    same=event["p_same"]
    f=crop_features(rgb,depth,box,ref,pose,estimate)
    n,inter,iou=box_grade(mask,box)
    # Change one data source at a time in the frozen model. These checks only
    # measure local sensitivity; they do not provide an implementable policy.
    oracle_count=dict(estimate,estimated_pixels=n)
    p_area=model.predict(rgb,depth,box,ref,pose,oracle_count)
    p_box=model.predict(rgb,depth,true_box,ref,pose,estimate)
    l,t,r,b=box
    return {**mask_measure(mask,depth), **source_support(mask,depth,pose,initial_depth,initial_mask,ref),
            "p_clear":predicted,"p_same":same,
            "coverage":inter/n,"iou":iou,"true_target_fraction_in_box":inter/((r-l)*(b-t)),
            "source_projection_px":float(estimate["estimated_pixels"]),"current_crop_estimated_target_px":float(f[8]),
            "actual_inside_box_px":int(inter),
            "projection_below_actual":estimate["estimated_pixels"]<n,
            "current_crop_estimate_below_actual_inside_box":float(f[8])<inter,
            "projection_on_wrong_side_of_cutoff":(estimate["estimated_pixels"]>=369)!=(n>=369),
            "perfect_projected_area_changes_accept":bool((p_area*same>=cutoff)!=(predicted*same>=cutoff)),
            "perfect_rect_changes_accept":bool((p_box*same>=cutoff)!=(predicted*same>=cutoff))}


def summarize(records):
    fields=defaultdict(list)
    for item in records:
        for k,v in item.items():
            fields[k].append(v)
    return {"groups":len(records),"measures":{k:(int(sum(v)) if isinstance(v[0],(bool,np.bool_)) else distribution(v)) for k,v in fields.items()}}


def main():
    groups=defaultdict(list)
    oracle=Counter()
    model=ClarityEstimator.load()
    for path in sorted((RUN/"private_paired/full").glob("*/*/decision.json")):
        house,src=path.parent.parent.name,path.parent.name
        current=json.loads(path.read_text())
        trace=current["trace"]
        older=json.loads((OLD/"private_roll"/house/(src+".json")).read_text())
        ot=older["new_confirmation"]
        old=older["events"]
        with np.load(RUN/"private_scoring"/house/(src+".npz")) as d:
            masks=np.unpackbits(d["packed"],axis=1).reshape(-1,192,192)
        with np.load(RUN/"private_inputs"/house/(src+".npz")) as d:
            rgb,depth,poses,source_box=(d[k] for k in ("rgb","depth","odom","cue"))
        prior_missed=ot is None and any(e["px"]>=369 for e in old)
        prior_unseen=ot is None and not any(e["px"]>=369 for e in old)
        if prior_unseen:
            sizes=masks.sum(axis=(1,2))
            t=int(np.argmax(sizes))
            m=mask_measure(masks[t],depth[t])
            assert m
            m["start_to_best_gain_in_pixels"]=int(sizes[t]-sizes[0])
            groups["previous_33_best_frame_on_full_route"].append(m)
        at=current["step"]
        if at is not None:
            good=eligible(masks[at],trace[at]["box"])
            if good:
                which="all_new_correct_first_submissions"
            elif (ot is None or old[ot]["correct"]) and 0<masks[at].sum()<369:
                which="five_new_early_small_target_reports"
            else:
                continue
        elif prior_missed:
            good=[i for i,e in enumerate(trace) if eligible(masks[i],e["box"])]
            # Counterfactual clarity-only oracle; the identity decision and
            # available predicted candidate remain the policy's own.
            possible=next((i for i,e in enumerate(trace) if e["box"] is not None and masks[i].sum()>=369 and e["p_same"]>=thresholds()["threshold"]),None)
            oracle["remaining_old_miss"]+=1
            if possible is not None:
                oracle["correct_if_perfect_area" if eligible(masks[possible],trace[possible]["box"]) else "wrong_box_if_perfect_area"]+=1
            else:
                oracle["still_abstain_if_perfect_area"]+=1
            if not good:
                continue
            at=max(good,key=lambda i:trace[i]["p_clear"]*trace[i]["p_same"])
            if trace[at]["p_clear"]>=.835 or trace[at]["p_same"]<.835:
                continue
            which="fourteen_clarity_only_missed_best_valid_frame"
        else:
            continue
        assert trace[at]["state"]=="trusted_initial"
        ref=FrameConfirmer(rgb[0],depth[0],source_box)
        groups[which].append(frozen_scores(rgb[at],depth[at],poses[at],masks[at],trace[at],ref,model,depth[0],masks[0]))

    assert sorted(map(len,groups.values()))==[5,14,30,33]
    assert oracle["remaining_old_miss"]==22
    out={"selection":"One frame per request: best joint score among spatially valid and actually clear frames for the 14; first submission for the 5 and 30; highest target-pixel count on the saved uninterrupted route for the 33. The subsets answer different questions.",
         "sensitivity":"Replace only the count estimated by source-surface reprojection (including derived ratios) OR the candidate rectangle used for current-crop features with its true envelope; keep all weights, native frame, source reference and p_same fixed. Flips are single-frame sensitivity, not navigation successes.",
         "visible_surface_depth":"Median measured optical-axis depth only on visible target pixels, not centroid distance, true object size or depth behind occluders.",
         "subsets":{name:summarize(rows) for name,rows in groups.items()},
         "perfect_area_first_event_on_remaining_old_misses":oracle}
    (HERE/"observed_clarity.json").write_text(json.dumps(out,indent=2)+"\n")
    for name,rows in groups.items():
        s=summarize(rows)["measures"]
        print(name,len(rows),{k:(v["median"] if isinstance(v,dict) else v) for k,v in s.items()})
    print(dict(oracle))


if __name__=="__main__":
    main()
