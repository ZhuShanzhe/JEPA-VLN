"""Independent instance-mask evaluator. Navigation predictions have all been written before this pass."""
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np

from probe_branches import HERE, ROOT, MV, physical, FAMILY, read, digest, select
from review_interface import box_match
from visibility_review import configured_sim, DATA, _load_goal_sets, load_scene_instances, instance_lut, reference_proximity
from run_multiview import environment


def main():
    private=HERE/"probe_private"
    all_rows=[]
    windows=[]
    distinct_by_window={w:set() for w in (4,12,24,48)}
    seen_signatures=set()
    for house,spec in enumerate(read(FAMILY/"panel.json")):
        path=private/f"branches_{house}.json"
        rows=read(path)
        prefix=[json.loads(x) for x in (MV/"private_events"/f"{house}.poses.jsonl").read_text().splitlines()]
        events=[json.loads(x) for x in (MV/"private_events"/f"{house}.jsonl").read_text().splitlines()]
        assert list(select(events)) and {r["step"] for r in rows}==set(select(events))
        ep,old=environment(spec)
        old.close()
        goal=ep["object_category"]
        classes=_load_goal_sets(DATA)["scoring_class_ids"]["hssdov"][goal]
        records=[r for r in load_scene_instances(DATA/"protocol/instance_viewpoints/hssd","val",ep["scene_id"])
                 if r["class_id"] in classes and r["view_points"]]
        targets=[r["semantic_id"] for r in records]
        sim=configured_sim(ep)
        lut=instance_lut(sim,"hssd",DATA/"vocab")
        objs=sim.get_rigid_object_manager()
        by_id={o.semantic_id:Path(o.creation_attributes.handle).name.removesuffix(".object_config.json")
               for o in (objs.get_object_by_handle(k) for k in objs.get_object_handles())}
        assert all(lut[r["semantic_id"]]==r["class_id"]+1 and by_id[r["semantic_id"]]==r["template_name"] for r in records)
        threshold=math.ceil(192*192*.01)
        def key(t):
            return tuple(round(float(x),6) for x in t["xyz"]+t["quat_xyzw"])
        def mask(t):
            cam=physical.hs.AgentState()
            cam.position=np.float32(t["xyz"])
            cam.rotation=physical.quat_from_coeffs(np.asarray(t["quat_xyzw"]))
            sim.get_agent(0).set_state(cam,reset_sensors=True)
            raw=np.asarray(sim.get_sensor_observations()["instance_mask"])
            return np.where(np.isin(raw,targets),raw,0)
        counts={}
        # Archive-only status of the same threshold over longer horizons. This
        # never enters the branch proposal or the frozen policy ranking.
        states=list(prefix[:min(len(prefix),max(r["step"] for r in rows)+49)])
        for r in rows:
            states.extend(t for a in r["outcomes"] for t in a["trail"])
        unique={key(t):t for t in states}
        try:
            for k,t in unique.items():
                ids,qty=np.unique(mask(t),return_counts=True)
                counts[k]={int(v):int(n) for v,n in zip(ids,qty) if v}
        finally:
            sim.close()
        close=dict(zip(unique,reference_proximity(ep,list(unique.values()),records)))
        source_expected=read(MV/"interface_review.json")["houses"][house]["selected_cues"]
        expected=iter(e for e in source_expected if e["anchored"] and not e["site_offered"])
        semsim=configured_sim(ep)
        try:
            # Habitat rigid objects must be assigned the same semantic IDs on each
            # newly created simulator, including this source-box audit renderer.
            box_lut=instance_lut(semsim,"hssd",DATA/"vocab")
            assert all(box_lut[r["semantic_id"]]==r["class_id"]+1 for r in records)
            for r in rows:
                mode=r["mode"]
                step=r["step"]
                seen={sid for t in prefix[:step+1] for sid,n in counts[key(t)].items() if n>=threshold}
                seen32={sid for t in prefix[:step+1] for sid,n in counts[key(t)].items() if n>=32}
                seen_any={sid for t in prefix[:step+1] for sid in counts[key(t)]}
                scanids={sid for t in prefix[step-4:step+1] for sid,n in counts[key(t)].items() if n>0}
                if mode=="discover":
                    item=dict(house=house+1,sample=step//100,complete={},first_instances_in_interval={},cumulative_instances={})
                    old_h=0
                    available=set(seen)
                    for h in (4,12,24,48):
                        complete=step+h<len(prefix)
                        item["complete"][h]=complete
                        if complete:
                            arrived={sid for t in prefix[step+old_h+1:step+h+1] for sid,n in counts[key(t)].items() if n>=threshold}
                            new=arrived-available
                            item["first_instances_in_interval"][h]=len(new)
                            available.update(new)
                            item["cumulative_instances"][h]=len(available-seen)
                            distinct_by_window[h].update((house,sid) for sid in new)
                        else:
                            item["cumulative_instances"][h]=None
                            item["first_instances_in_interval"][h]=None
                        old_h=h
                    windows.append(item)
                box=r["box"]
                target=None
                source_px=None
                valid=None
                if box:
                    s=prefix[step-(3,4,1)[box["view_index"]]]
                    cam=physical.hs.AgentState()
                    cam.position=np.float32(s["xyz"])
                    cam.rotation=physical.quat_from_coeffs(np.asarray(s["quat_xyzw"]))
                    semsim.get_agent(0).set_state(cam,reset_sensors=True)
                    raw=semsim.get_sensor_observations()["instance_mask"]
                    sid,iou,_=box_match(np.where(np.isin(raw,targets),raw,0),box["bbox_norm"])
                    valid=bool(sid and iou>=.25)
                    frozen_cue=next(expected)
                    assert frozen_cue["valid_box"]==valid,(house,step,box["view_index"],sid,iou,frozen_cue)
                    if valid:
                        target=sid
                        source_px=int((raw==target).sum())
                outcomes={}
                for a in r["outcomes"]:
                    frames=[counts[key(t)] for t in a["trail"]]
                    area=[f.get(target,0) for f in frames] if valid else []
                    unseen=set()
                    emerging32=set()
                    emerging_any=set()
                    history=[]
                    grounded=set()
                    for f,t in zip(frames,a["trail"]):
                        visible={sid for sid,n in f.items() if n>=threshold}
                        unseen.update(visible-seen)
                        emerging32.update({sid for sid,n in f.items() if n>=32}-seen32)
                        emerging_any.update(set(f)-seen_any)
                        grounded.update(visible&close[key(t)])
                        history.append(len(unseen))
                    outcomes[a["id"]]=dict(length=len(frames),planned=len(a["actions"]),collision=a["collision"],
                        surface_gain=a["surface_gain"],new_instances=len(unseen),new_by_action=history,
                        new_instances_32px=len(emerging32),new_instances_any_pixel=len(emerging_any),
                        close_and_visible=len(grounded),same_cue_close_visible=(target in grounded) if valid else None,
                        cue_final=area[-1] if area else None,
                        cue_best=max(area) if area else None,cue_by_action=area)
                model=outcomes[r["selected"]]
                geom=outcomes[r["geometry"]]
                base=outcomes["continue"]
                base_steps=next(a["trail"] for a in r["outcomes"] if a["id"]=="continue")
                replayed_base_matches_host=all(np.allclose(t["xyz"],p["xyz"],atol=1e-5,rtol=0)
                    and abs(np.dot(t["quat_xyzw"],p["quat_xyzw"]))>1-1e-6
                    for t,p in zip(base_steps,prefix[step+1:step+1+len(base_steps)]))
                def compact(a):
                    return {k:v for k,v in a.items() if k not in ("cue_by_action","new_by_action")}
                scores=[q["model_score"] for q in r["outcomes"] if q["model_score"] is not None]
                changed=len(scores)>1
                trials=len(outcomes)
                item=dict(house=house+1,goal=goal,mode=mode,sample=step//100 if mode=="discover"
                    else sum(1 for v in all_rows if v["house"]==house+1 and v["mode"]=="verify")+1,
                    gate=r["gate"],proposal=r["proposal"],choices=trials,ranked=changed,
                    native_regional_trajectory_matches_prefix=bool(replayed_base_matches_host),
                    inspection_valid=valid,initial_cue_pixels=source_px,
                    initial_scan_target_pixels_present=bool(scanids),
                    selected_continue=r["selected"]=="continue",geometry_continue=r["geometry"]=="continue",
                    matches_geometry=r["selected"]==r["geometry"],
                    distinct_model_scores=len({round(float(s),8) for s in scores}),
                    best_new_instances=max(a["new_instances"] for a in outcomes.values()),
                    best_new_32px=max(a["new_instances_32px"] for a in outcomes.values()),
                    best_new_any_pixel=max(a["new_instances_any_pixel"] for a in outcomes.values()),
                    best_same_cue_close_visible=any(a["same_cue_close_visible"] for a in outcomes.values()) if valid else None,
                    best_surface_gain=max(a["surface_gain"] for a in outcomes.values()),
                    best_final_cue_pixels=max(a["cue_final"] for a in outcomes.values()) if valid else None,
                    best_close_visible=max(a["close_and_visible"] for a in outcomes.values()),
                    total_candidate_actions=sum(a["length"] for a in outcomes.values()),
                    candidate_collisions=sum(a["collision"] for a in outcomes.values()),
                    model=compact(model),geometry=compact(geom),regional=compact(base))
                # Matched-action comparison with the regional prefix. The ceiling
                # above is opportunity under the common cap, not a realizable policy.
                for name,a in (("model",model),("geometry",geom)):
                    n=min(base["length"],a["length"])
                    item[name]["paired_actions"]=n
                    item[name]["paired_new_delta"]=a["new_by_action"][n-1]-base["new_by_action"][n-1]
                    item[name]["paired_cue_pixel_delta"]=a["cue_by_action"][n-1]-base["cue_by_action"][n-1] if valid else None
                assert (house,step) not in seen_signatures
                seen_signatures.add((house,step))
                all_rows.append(item)
        finally:
            semsim.close()
        assert next(expected,None) is None
        print("posthoc scored",house,len(rows),flush=True)
    sums={}
    groups=dict(discovery=[r for r in all_rows if r["mode"]=="discover"],
                source_verified=[r for r in all_rows if r["inspection_valid"]],
                source_incorrect=[r for r in all_rows if r["inspection_valid"] is False])
    assert len(groups["source_verified"])==len(groups["source_incorrect"])==4
    for label,rows in groups.items():
        stats=dict(cases=len(rows),choice_available=sum(r["choices"]>1 for r in rows),ranked=sum(r["ranked"] for r in rows),
            switched=sum(not r["selected_continue"] for r in rows),cases_with_absent_target_in_all_source_views=sum(not r["initial_scan_target_pixels_present"] for r in rows),
            capped_candidate_new_instances=sum(r["best_new_instances"] for r in rows),
            capped_candidate_new_32px=sum(r["best_new_32px"] for r in rows),
            capped_candidate_new_any_pixel=sum(r["best_new_any_pixel"] for r in rows),
            capped_candidate_close_visible=sum(r["best_close_visible"] for r in rows),
            candidate_routes=sum(r["choices"] for r in rows),physical_actions=sum(r["total_candidate_actions"] for r in rows),
            candidate_collisions=sum(r["candidate_collisions"] for r in rows),
            regional_prefix_agrees_with_archived_poses=sum(r["native_regional_trajectory_matches_prefix"] for r in rows),
            gate_counts=dict(Counter(r["gate"] for r in rows)))
        for name in ("regional","geometry","model"):
            stats[name]={k:sum(r[name][k] for r in rows) for k in ("length","surface_gain","new_instances","close_and_visible","collision")}
        sums[label]=stats
    report=dict(selection="First entered regional explore with >=4 regional actions per house per [0,99],[100,199],[200,295], plus all 8 projection-without-site cues from archived evaluation; no target labels used to select states.",
        meaning="Independent forks at fixed archived states. Four actions maximum. Surface gain is newly measured 0.4m obstacle-return bins; new instance means first observation at >=1% of frame relative to whole recorded prefix. Near uses original 0.25m reference-viewpoint convention. Outcomes are per probe, not house-level distinct reports. Gating cooldown starts fresh for each fork. Blocked prefix stops on contact.",
        paid_calls=0,source_hashes={name:digest(HERE/name) for name in ("routes.py","gateway.py","probe_branches.py","probe_score.py")},
        summary=sums,rows=all_rows)
    (HERE/"probe_results.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    window_summary={}
    for horizon in (4,12,24,48):
        eligible=[r for r in windows if r["complete"][horizon]]
        hits=[r for r in eligible if r["first_instances_in_interval"][horizon]]
        window_summary[horizon]=dict(eligible=len(eligible),houses_observed=len({r["house"] for r in eligible}),
            states_with_first_in_interval=len(hits),first_instances_in_interval=sum(r["first_instances_in_interval"][horizon] for r in hits),
            different_house_instance_pairs_in_interval=len(distinct_by_window[horizon]),houses_gaining_in_interval=len({r["house"] for r in hits}),
            cumulative_states_with_any_new=sum(bool(r["cumulative_instances"][horizon]) for r in eligible),
            cumulative_instances_per_state_sum=sum(r["cumulative_instances"][horizon] for r in eligible))
    window_report=dict(threshold="First time the same scored instance occupies at least one percent of a 192x192 central frame since episode start.",
        source="Physically recorded original three-view HSSD navigation trajectories; camera turns/scans count towards elapsed actions. No new policy decisions or rollouts.",
        protocol="Same 24 discovery probes, each fixed before semantic scoring; intervals 1-4, 5-12, 13-24, 25-48 relative to the fork. Each complete window counts once per probe, with three probes drawn from each house. Distinct house/instance deduplicates physical IDs across starting points.",
        evaluator_sha256=digest(HERE/"probe_score.py"),summary=window_summary,rows=windows)
    (HERE/"probe_windows.json").write_text(json.dumps(window_report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(sums),flush=True)


if __name__=="__main__":
    main()
