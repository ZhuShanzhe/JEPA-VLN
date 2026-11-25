"""Post-episode Habitat audit: correlate saved observations, report poses and official task targets."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import numpy as np

def read(path):
    return json.loads(path.read_bytes())


def audit(project, row, ref, oracle_replay, native=False):
    from check_routes import semantic_scene
    from evaluation.main.evaluator import _reset_environment, _env_action, _near_target_ids
    from verify_habitat import state, digest
    file=Path(read(ref)["result"])
    receipt=read(oracle_replay)
    assert digest(file)==receipt["source_result_sha256"]==read(ref)["result_sha256"]
    doc=read(file)
    ep,=doc["episodes"]
    trace=ep["paper"]["steps"]
    action_trace=[json.loads(x) for x in (file.parent/"agent/action_trace.jsonl").read_bytes().splitlines()]
    assert [(r["step"],r["action"]) for r in trace]==[(r["step"],r["actual_action"]["id"]) for r in action_trace]
    confirm_path=file.parent/"agent/confirm_events.jsonl"
    memory=[] if native or not confirm_path.is_file() else [json.loads(x) for x in confirm_path.read_bytes().splitlines()]
    assigned=defaultdict(list)
    for i,t in enumerate(action_trace):
        if t["mapping_frame_id"] is not None:
            assigned[t["mapping_frame_id"]].append(i)
    centers=defaultdict(list)
    projected=defaultdict(list)
    target_crop_fractions=[]
    for idx,event in enumerate(memory):
        if event["event"]=="view_saved" and event.get("center_1000") is not None:
            for t in assigned.get(event.get("frame"),[]):
                if t<=event["step"]:
                    centers[t].append((idx,event["center_1000"]))
    env,item,data=semantic_scene(doc)
    threshold=doc["manifest"]["arguments"]["threshold"]
    report={r["step"]:r for r in ep["report_events"]}
    target_ids=set(map(int,item.target_object_ids))
    opportunities=set()
    simultaneously=set()
    credited=set()
    last_in_range={}
    near_steps=[]
    clearly_visible_near_steps=[]
    per_step=[]
    notices=[]
    try:
        frame=_reset_environment(env,item,data)
        for i,t in enumerate(trace):
            pos=tuple(state(env)[0])
            assert np.linalg.norm(np.array(pos)-t["position_before"])<=.002
            ids=set(map(int,_near_target_ids(pos,item.target_object_ids,item.target_positions,
                        threshold=threshold,pathfinder=env.sim.pathfinder,target_view_points=item.view_points)))
            sem=np.asarray(frame["semantic"]).squeeze()
            assert sem.shape==(320,320)
            pixels=np.bincount(sem.ravel().clip(min=0).astype(np.int32))
            for idx,xy in centers[i]:
                x,y=np.rint(np.asarray(xy)*319/1000).astype(int)
                assert 0<=x<320 and 0<=y<320
                projected[idx].append(int(sem[y,x]) in target_ids)
                region=sem[max(0,y-14):min(320,y+15),max(0,x-14):min(320,x+15)]
                target_crop_fractions.append(float(np.isin(region,list(target_ids)).mean()))
            seen={k for k in target_ids if k<len(pixels) and pixels[k]>=103}
            fresh=ids-credited
            for k in fresh:
                last_in_range[k]=i
            opportunities.update(fresh)
            simultaneously.update(fresh & seen)
            if fresh:
                near_steps.append(i)
            if fresh & seen:
                clearly_visible_near_steps.append(i)
            per_step.append((frozenset(ids),frozenset(seen)))
            if i in report:
                r=report[i]
                assert ids==set(map(int,r["near_targets"]))
                notices.append(dict(step=i,nearest_valid_m=round(float(r["report_ne"]),3)
                                  if r["report_ne"] is not None and np.isfinite(r["report_ne"]) else None,
                                  previously_uncredited_in_range=len(fresh), newly_credited=len(r["new_targets"]),
                                  clear_target_in_current_view=bool(seen),
                                  uncredited_target_both_clear_and_in_range=bool(fresh & seen),
                                  steps_since_last_chance=min((i-v for k,v in last_in_range.items() if k not in credited),default=None)))
                credited.update(map(int,r["new_targets"]))
            if t["action_role"]=="navigate":
                frame=env.step(_env_action(t["action"]))
    finally:
        env.close()
    assert credited==set(map(int,ep["reached_targets"]))
    memory_summary=Counter()
    for idx,e in enumerate(memory):
        if e["event"]!="view_saved":
            continue
        matches=[s for s in assigned.get(e.get("frame"),[]) if s<=e["step"]]
        memory_summary["saved_events"]+=1
        if not matches:
            memory_summary["unmatched_source_frame"]+=1
            continue
        # Count only invariant membership when a mapping keyframe was reused across steps.
        visible=[bool(per_step[s][1]) for s in matches]
        nearby=[bool(per_step[s][0]) for s in matches]
        memory_summary["source_frames_target_visible_consistently"]+=all(visible)
        memory_summary["source_frames_no_visible_target"]+=not any(visible)
        memory_summary["source_frames_any_valid_target_near_consistently"]+=all(nearby)
        if projected[idx]:
            memory_summary["center_on_task_target_in_all_mapped_frames"]+=all(projected[idx])
            memory_summary["center_outside_task_targets_in_all_mapped_frames"]+=not any(projected[idx])
    if target_crop_fractions:
        memory_summary["min_target_pixel_fraction_in_replayed_template"]=round(min(target_crop_fractions),4)
        memory_summary["max_target_pixel_fraction_in_replayed_template"]=round(max(target_crop_fractions),4)
    # Only count post-action observations for steps actually executed as selected.
    # A changed or interrupted command can yield new evidence on its own.
    scheduled_frames=set()
    routes=file.parent/"agent/v13_routes.jsonl"
    if not native and routes.is_file():
        entries=[json.loads(x) for x in routes.read_bytes().splitlines()]
        ends={e["start"]:e for e in entries if e["event"]=="view_track_end"}
        for e in entries:
            if e["event"]!="view_track_start":
                continue
            start=e["step"]
            length=ends.get(start,{}).get("confirmed_commands",0)
            assert 0<=length<=len(e["prefix"])
            assert [s["action"] for s in trace[start:start+length]]==e["prefix"][:length]
            scheduled_frames.update(range(start+1,min(start+length+1,len(per_step))))
    first_clear={}
    for i,(_,visible) in enumerate(per_step):
        for instance in visible:
            first_clear.setdefault(instance,i)
    return dict(distinct_previously_uncredited_ever_in_range=len(opportunities),
                distinct_previously_uncredited_in_range_and_clear_same_frame=len(simultaneously),
                steps_with_uncredited_target_in_range=len(near_steps),
                steps_with_uncredited_target_clear_and_in_range=len(clearly_visible_near_steps),
                first_step_clear_and_in_range=next(iter(clearly_visible_near_steps),None),
                last_step_clear_and_in_range=clearly_visible_near_steps[-1] if clearly_visible_near_steps else None,
                unique_instances_ever_clear_in_one_frame=len(first_clear),
                observations_after_acknowledged_selected_commands=len(scheduled_frames) if not native else None,
                first_clear_exposures_during_acknowledged_commands=sum(t in scheduled_frames for t in first_clear.values()) if not native else None,
                report_checks=notices,saved_view_checks=dict(memory_summary),
                confirmation_event_file_present=confirm_path.is_file(),summary_events=dict(Counter(e["event"] for e in memory)),
                reason_counts=dict(Counter(str(e["reason"]) for e in memory if e.get("reason"))))


def main(project, output, baseline_only=False):
    sys.path.insert(0,str(project/"jepa_completion_v17/physical"))
    from verify_rank import deny_ip_sockets
    deny_ip_sockets()
    sys.path.insert(0,str(project/"jepa_confirm_v18/agent"))
    from view_receipt import run as certify_view
    pilot=project/"mapping_runtime/pilots/jepa_confirm_v18_first2"
    rows=[]
    for r in read(pilot/"panel.json")["rows"]:
        file=r["episode_id"]+".json"
        b=audit(project,r,Path(r["ref"]), project/"mapping_runtime/analysis/visibility/native_20261115/verified"/file,True)
        result=dict(ordinal=r["ordinal"],native=b)
        if not baseline_only:
            ref=pilot/"comparison/accepted/second"/file
            result["treatment"]=audit(project,r,ref,pilot/"replay"/file)
            result["view_execution"]=certify_view(project,Path(read(ref)["result"]))
        rows.append(result)
    output.write_text(json.dumps(rows,allow_nan=False,indent=2)+"\n")


if __name__=="__main__":
    main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]),"--native" in sys.argv[3:])
