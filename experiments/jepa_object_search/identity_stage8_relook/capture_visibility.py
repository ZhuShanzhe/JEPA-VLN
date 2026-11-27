"""Post-run truth check on every weak-clue capture, keeping only aggregate overlaps."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from audit_clock import PROJECT, STAGE8, accepted, digest, lines, load
from check_routes import semantic_scene
from evaluation.main.evaluator import _env_action, _near_target_ids, _reset_environment
from proposal_worker import ProposalWorker
from trusted_reference import MIN_DETECTOR_SCORE
from verify_rank import deny_ip_sockets


def signature(rgb):
    return np.asarray(Image.fromarray(np.asarray(rgb,dtype=np.uint8)[:,:,:3]).resize(
        (64,64),Image.Resampling.BOX),dtype=float)


def analyze_boxes(rgb,mask,boxes,target_names):
    h,w = mask.shape
    results=[]
    for det in boxes:
        score=float(det["score"])
        if score < MIN_DETECTOR_SCORE:
            continue
        scaled=np.asarray(det["bbox"],float)*np.tile([w/rgb.shape[1],h/rgb.shape[0]],2)
        l,t=np.clip(np.floor(scaled[:2]).astype(int),[0,0],[w,h])
        r,b=np.clip(np.ceil(scaled[2:]).astype(int),[l,t],[w,h])
        crop=mask[t:b,l:r]
        cx,cy=np.clip(((scaled[:2]+scaled[2:])/2).astype(int),[0,0],[w-1,h-1])
        by_target={}
        for tid,name in target_names.items():
            hits=int(np.count_nonzero(crop==tid))
            if hits:
                count=int(np.count_nonzero(mask==tid))
                by_target[name]=dict(overlapping_pixels=hits,fraction_of_box=round(hits/crop.size,4),
                                     fraction_of_visible_object=round(hits/count,4))
        results.append(dict(score=round(score,4),box_pixels_at_320=int(crop.size),
                            center_on_target=target_names.get(int(mask[cy,cx])),intersections=by_target))
    return sorted(results,key=lambda p:-p["score"])


def main():
    deny_ip_sockets()
    assert (STAGE8/"closed.json").is_file()
    row,=load(STAGE8/"panel.json")["rows"]
    eid=row["episode_id"]
    ticket=STAGE8/"comparison/accepted/second"/(eid+".json")
    path,doc,ep=accepted(ticket)
    assert digest(path)==load(STAGE8/"score_receipts"/(eid+".json"))["source_result_sha256"]
    _,_,native=accepted(row["ref"])
    credited_native=list(map(int,native["reached_targets"]))
    assert len(credited_native)==1
    folder=path.parent/"agent"
    notes=lines(folder/"confirm_events.jsonl")
    actions=lines(folder/"action_trace.jsonl")
    det_logs=lines(folder/"multiview_events.jsonl")
    cats={q["category"] for d in lines(folder/"decision_trace.jsonl")
          if (q:=d["world_state"].get("local_rgb_proposals")) and q["category"]}
    assert len(cats)==1
    category=cats.pop()
    captures={r["step"]:r for r in notes if r["event"]=="weak_relook_capture"}
    assert len(captures)==sum(r["event"]=="weak_relook_capture" for r in notes)
    rt=STAGE8.parents[1]
    service=rt/"runs"/load(STAGE8/"mapper_process.json")["run"]
    frame_src=service/"mapping/diagnostics"/(eid+"_frames")
    env,item,data=semantic_scene(doc)
    worker=None
    results=[]
    target_ids=sorted(map(int,item.target_object_ids))
    others=[i for i in target_ids if i not in credited_native]
    lookup={credited_native[0]:"original_baseline_credited_clock",
            **{t:"other_eligible_clock_"+str(i+1) for i,t in enumerate(others)}}
    threshold=doc["manifest"]["arguments"]["threshold"]
    max_error=0.
    try:
        if captures:
            worker=ProposalWorker(PROJECT/"b_verification/env/bin/python",PROJECT,"cuda")
        frame=_reset_environment(env,item,data)
        for step,original in enumerate(ep["paper"]["steps"]):
            if not captures or step>max(captures):
                break
            state=env.sim.get_agent_state()
            err=float(np.linalg.norm(state.position-original["position_before"]))
            max_error=max(err,max_error)
            assert err < .002
            if step in captures:
                observed=captures[step]
                action=actions[step]
                fid=observed["frame"]
                rgb=np.ascontiguousarray(frame["rgb"][:,:,:3])
                mask=np.asarray(frame["semantic"]).squeeze()
                assert mask.shape==(320,320)
                sem,rgb_pose=(state.sensor_states[s] for s in ["semantic","rgb"])
                assert np.allclose(sem.position,rgb_pose.position,atol=1e-6)
                assert abs(abs(np.dot(sem.rotation.components,rgb_pose.rotation.components))-1)<1e-5
                filename=frame_src/("rgb_%06d.jpg"%fid)
                with Image.open(filename) as im:
                    recorded=np.asarray(im.convert("RGB"))
                mae=float(np.abs(signature(recorded)-signature(rgb)).mean())
                near=set(map(int,_near_target_ids(state.position,item.target_object_ids,item.target_positions,
                    threshold=threshold,pathfinder=env.sim.pathfinder,target_view_points=item.view_points)))
                previous_query=[r for r in det_logs if r["event"]=="boxes" and r["source"]=="live"
                                and r["step"]==step and r["frame"]==step]
                assert previous_query and not any(r["worker_failed"] for r in previous_query)
                original_counts={r["count"] for r in previous_query}
                assert len(original_counts)==1
                rerun=worker.detect(rgb,category)
                decoded=worker.detect(recorded,category)
                aligned=(mae<=3.0 and fid==action["mapping_frame_id"])
                record=dict(step=step,capture_equals_live_frame=fid==action["mapping_frame_id"],
                    rgb_mae_at_64=round(mae,4),rgb_matches_capture=aligned,capture_jpg_sha256=digest(filename),
                    stored_total_proposal_count=original_counts.pop(),rerun_total_proposal_count=len(rerun),
                    visible_pixels={name:int(np.count_nonzero(mask==t)) for t,name in lookup.items()},
                    in_official_range=sorted(lookup[t] for t in near),
                    live_rgb_high_confidence=analyze_boxes(rgb,mask,rerun,lookup),
                    recorded_jpeg_high_confidence=analyze_boxes(recorded,mask,decoded,lookup) if aligned else None,
                    geometry_result_available=any(r["event"]=="weak_relook_evidence" and r.get("frame")==fid
                                                   for r in notes))
                results.append(record)
            if original["action_role"]=="navigate":
                frame=env.step(_env_action(original["action"]))
    finally:
        if worker: worker.close()
        env.close()
    result=dict(source_result_sha256=digest(path),selection="All weak_relook_capture events in the accepted run",
                target_labels="Stable anonymized labels relative to the clock credited in the original MOST-Nav episode",
                box_source="All live boxes above the original detector confidence gate from a rerun on the reproduced lossless RGB, and a second detection on its stored JPEG. The online trace archived counts, not the live coordinates at every step. No image or box coordinates exported.",
                semantic_resolution=[320,320],online_detector_query=category,detector_minimum_confidence=MIN_DETECTOR_SCORE,
                official_reporting_distance_m=threshold,max_replay_position_error_m=max_error,
                capture_count=len(results),paid_calls=0,results=results)
    dest=Path(__file__).with_name("capture_visibility.json")
    if dest.exists():
        assert load(dest)==result
    else:
        dest.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result["results"],allow_nan=False))


if __name__=="__main__":
    main()
