"""Post-hoc decomposition on the frozen blind trajectories. Never changes inference."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from frame_confirm import FrameConfirmer, thresholds
from rollout import ACTION, ASSETS, SOURCE_H, camera, inventory, simulator, OLD, HERE


def overlap(e):
    return e["box"] is not None and e["intersection"] >= .5*e["px"] and e["gt_box_iou"] >= .3


def score_gate(e, threshold):
    clear, same=e["p_clear"],e["p_same"]
    if clear < threshold and same < threshold: return "both_below_product_cutoff"
    if clear < threshold: return "clarity_alone_below_product_cutoff"
    if same < threshold: return "identity_alone_below_product_cutoff"
    return "both_individually_above_product_cutoff_but_product_below"


def projection(trace, src_mask, obs, pose, oid):
    # Ordinary production gates; ground truth below only counts point identities at either end.
    x,z,yaw=pose
    px,py,pz=trace.xyz.T
    dx,dz=px-x,pz-z
    fw=-dx*np.sin(yaw)+dz*np.cos(yaw)
    right=dx*np.cos(yaw)+dz*np.sin(yaw)
    uf=96+96*right/np.maximum(fw,.01)
    vf=96-96*py/np.maximum(fw,.01)
    field=(fw>.12)&(uf>=0)&(uf<192)&(vf>=0)&(vf<192)
    ok=field&(trace.w>.12)
    ui,vi=uf[ok].astype(int),vf[ok].astype(int)
    dep=obs["depth"].squeeze();rgb=obs["rgb"][...,:3].astype(np.float32)
    projected=fw[ok]
    def residual(i,j):
        u=(ui+i).clip(0,191);v=(vi+j).clip(0,191)
        return abs(dep[v,u]-projected),np.mean(abs(trace.rgb[ok]-rgb[v,u]),axis=1)
    d,c=residual(0,0); loss=d/(.045+.025*projected)+c/42
    for off in ((0,1),(1,0),(-1,0),(0,-1)):
        dd,cc=residual(*off);cost=dd/(.045+.025*projected)+cc/42
        improve=cost<loss;loss=np.minimum(loss,cost)
        d,c=np.where(improve,dd,d),np.where(improve,cc,c)
    eligible=trace.w[ok]>=.35
    depth_ok=d<.07+.045*projected
    support=eligible & depth_ok & (c<68)
    target_end=obs["ids"][vi,ui]==oid
    return dict(
        source_support=int((trace.w>=.35).sum()), source_support_true=int((src_mask & (trace.w>=.35)).sum()),
        in_view=int(eligible.sum()),depth_supported=int((eligible&depth_ok).sum()),
        surface_supported=int(support.sum()),supported_true_at_source=int((support & src_mask[ok]).sum()),
        supported_true_at_destination=int((support & target_end).sum()),
        supported_true_at_both=int((support & src_mask[ok] & target_end).sum()),
    )


def quantiles(vals):
    return {k:round(float(v),3) for k,v in zip(("min","q25","median","q75","max"),np.quantile(vals,[0,.25,.5,.75,1]))} if vals else {}


def main():
    cutoff=float(thresholds()["threshold"])
    houses=json.loads((OLD/"view_design.json").read_text())["cohorts"]["blind"]
    groups=[]; errors=[]
    for house in houses:
        records=sorted((HERE/"private_roll"/house).glob("*.json"))
        assert len(records)==60
        sim=simulator(house)
        try:
            categories={sid:cat for cat,objects in inventory(sim,ASSETS).items() for sid,_,_ in objects}
            for log in records:
                r=json.loads(log.read_text())
                if not r["focus"]:continue
                events=r["events"];first=r["new_confirmation"]
                missed=first is None and any(e["px"]>=369 for e in events)
                empty_box=first is not None and events[first]["intersection"]==0
                if not (missed or empty_box):continue
                with np.load(OLD/"private_render_blind60"/("blind_"+house)/(str(r["source"])+"_sensor.npz")) as d:
                    start,yaw,oid,bb=d["position"],float(d["yaw"]),int(d["oid"]),d["box"]
                obs=camera(sim,start,yaw,SOURCE_H,0.)
                trace=FrameConfirmer(obs["rgb"],obs["depth"],bb)
                l,t,rr,b=bb; source_ids=obs["ids"][t:b,l:rr].ravel().copy()
                source_ids=source_ids[::max(1,int(np.ceil(len(source_ids)/6000)))]
                src_mask=source_ids==oid
                source_inside=int((obs["ids"][t:b,l:rr]==oid).sum())
                source_pixels=int((obs["ids"]==oid).sum())
                source_box_area=int((rr-l)*(b-t))
                clear_events=[]
                for e in events[:first+1] if first is not None else events:
                    if e["t"]:
                        sim.get_agent(0).act(ACTION[e["action"]]);obs=sim.get_sensor_observations()
                    if (empty_box and e["t"]!=first) or (missed and e["px"]<369):continue
                    ids=obs["ids"]
                    assert int((ids==oid).sum())==e["px"]
                    ver=trace.observe(obs["rgb"],obs["depth"],e["pose"])
                    assert (list(ver["box"]) if ver["box"] is not None else None)==e["box"]
                    assert abs(ver["p_clear"]-e["p_clear"])<1e-6 and abs(ver["p_same"]-e["p_same"])<1e-6
                    p=projection(trace,src_mask,obs,e["pose"],oid)
                    assert (p["surface_supported"]>=5)==ver["candidate"]
                    entry=dict(**e,proj=p,overlap=overlap(e),product=e["p_clear"]*e["p_same"])
                    clear_events.append(entry)
                    if empty_box:
                        l,t,rr,b=e["box"]; vals,cnt=np.unique(ids[t:b,l:rr],return_counts=True)
                        composition=[]
                        for i in np.argsort(-cnt)[:4]:
                            sid=int(vals[i])
                            composition.append(dict(label=categories.get(sid,"static_or_background" if sid<=0 else "other_geometry"),
                                                    pixels=int(cnt[i]),same_category=(categories.get(sid)==categories.get(oid))))
                        support_ids, support_counts=np.unique(source_ids[trace.w>=.35],return_counts=True)
                        source_support_classes=[dict(label=categories.get(int(support_ids[j]),"unclassified_geometry"),
                                                     points=int(support_counts[j])) for j in np.argsort(-support_counts)[:4]]
                        errors.append(dict(step=e["t"],source_category=categories[oid],source_pixels=source_pixels,
                          true_pixels_inside_source_box=source_inside,source_support_classes=source_support_classes,
                          source_box_area=source_box_area,current_target_pixels=e["px"],intersection=e["intersection"],
                          estimated_pixels=e["estimated_pixels"],box_area=(rr-l)*(b-t),
                          p_clear=e["p_clear"],p_same=e["p_same"],projection=p,
                          most_frequent_semantics_in_crop=composition))
                if missed: groups.append(clear_events)
        finally:sim.close()
    assert len(groups)==29 and len(errors)==1
    frames=[f for g in groups for f in g]
    def category(f):
        if f["overlap"]:return "candidate_meets_spatial_rule"
        if f["box"] is None:return "no_candidate"
        if f["intersection"]==0:return "candidate_empty_of_target"
        return "candidate_partial_or_poor_IoU"
    group_stage=Counter(); cause=Counter(); null_reasons=Counter()
    best_per_group=[];bad_box=[]
    for group in groups:
        good=[e for e in group if e["overlap"]]
        best=(max(good,key=lambda e:e["product"]) if good else None)
        if best:
            group_stage["at_least_one_spatially_acceptable_box"]+=1
            cause[score_gate(best,cutoff)]+=1;best_per_group.append(best)
        elif any(e["box"] is not None for e in group):
            group_stage["box_exists_but_never_spatially_acceptable"]+=1
            bad_box.append(max((e for e in group if e["box"]),key=lambda e:e["intersection"]/e["px"]))
        else:group_stage["no_box_on_any_clear_frame"]+=1
    for e in frames:
        p=e["proj"]
        if e["box"] is not None:continue
        if p["source_support"]<5:null_reasons["too_few_initial_weighted_surface_points"]+=1
        elif p["in_view"]<5:null_reasons["too_few_projected_points_in_view"]+=1
        elif p["depth_supported"]<5:null_reasons["too_few_depth_consistent_points"]+=1
        else:null_reasons["too_few_after_photometric_check"]+=1
    def statistics(es):
        return dict(n=len(es),estimated_pixels=quantiles([e["estimated_pixels"] for e in es]),
            true_pixels=quantiles([e["px"] for e in es]),p_same=quantiles([e["p_same"] for e in es]),
            p_clear=quantiles([e["p_clear"] for e in es]),product=quantiles([e["product"] for e in es]),
            coverage=quantiles([e["intersection"]/e["px"] for e in es]),
            support_correct_on_both_ends=quantiles([e["proj"]["supported_true_at_both"]/max(1,e["proj"]["surface_supported"]) for e in es]))
    good_frames=[e for e in frames if e["overlap"]]
    result=dict(group_stage=group_stage,frame_stage=Counter(category(e) for e in frames),
      no_box_frame_first_failing_gate=null_reasons,best_usable_box_first_failing_scores=cause,
      score_pattern_all_usable_frames=Counter(score_gate(e,cutoff) for e in good_frames),
      best_usable_box_per_group=statistics(best_per_group),best_coverage_of_bad_boxes_per_group=statistics(bad_box),
      all_usable_frames=statistics(good_frames),empty_box_false_confirmation=errors[0],
      limits="All recurrent frames counted, including turns and repeated poses. Groups are the primary unit. With existing boxes and trajectories, selecting a ground-truth-valid frame is an oracle ceiling. Scores are separately compared to the frozen product cutoff as a necessary, not sufficient, condition. No blind threshold sweep or retraining.",
      threshold=cutoff)
    out=HERE/"missed_confirmation_diagnosis.json"
    out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))

if __name__=="__main__":main()
