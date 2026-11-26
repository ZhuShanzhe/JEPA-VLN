"""Account for every registered recheck in both accepted V24 episodes. Private outputs stay on server."""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


P = Path(os.environ["NAV_ROOT"])
BASE = P / "identity_stage4_geometry/attribution"
PRIVATE = BASE / "private"


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(row) for row in path.read_bytes().splitlines() if row]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    from verify_views import semantic_scene, state
    from evaluation.main.evaluator import _env_action, _reset_environment, _near_target_ids
    from confirm_view import pose_after

    pilot = P / "mapping_runtime/pilots/jepa_reid_v24_first2"
    PRIVATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    summary = {}
    for idx,name in enumerate(("clock","painting")):
        eid = read(pilot / "panel.json")["rows"][idx]["episode_id"]
        ticket = read(pilot / "comparison/accepted/second" / (eid+".json"))
        path = Path(ticket["result"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ticket["result_sha256"]
        doc = read(path)
        ep, = doc["episodes"]
        times = ep["paper"]["steps"]
        hist = ep["paper"]["agent_trace"]
        agents = path.parent / "agent"
        trace = {row["step"]:row for row in lines(agents/"action_trace.jsonl")}
        assert list(trace) == list(range(300))
        event_file=agents/"confirm_events.jsonl"
        events = lines(event_file) if event_file.exists() else []
        if not event_file.exists():
            assert not hist["final_memory"]["instances"]
        rechecks = [r for r in events if r["event"]=="confirm_view_recheck"]
        kfs = [t for t,row in trace.items() if (row.get("mapping") or {}).get("is_keyframe")]
        npzs = list(path.parent.rglob("*.npz"))
        geo_keys = set()
        for f in npzs:
            with np.load(f,allow_pickle=False) as z:
                geo_keys.update(k for k in z.files if k in {"points","frame_points","depth","c2w","intrinsics","camera_points"})
        state_summary = dict(action_frames=len(trace), map_keyframes=len(kfs), map_nonkeyframes=len(trace)-len(kfs),
                             archived_geometry_field_names=sorted(geo_keys), final_memory_nodes=len(hist["final_memory"]["instances"]),
                             recheck_reasons=dict(Counter(r["reason"] for r in rechecks)), completed_rechecks=[])
        summary[name] = state_summary
        cases = []
        steps = set()
        for ordinal,attempt in enumerate(rechecks):
            if attempt["reason"]!="no_current_instance_match":
                continue
            end = attempt["step"]
            begin = next(e for e in reversed(events[:events.index(attempt)]) if e["event"]=="confirm_view_start"
                         and (e["instance"],e["frame"])==(attempt["instance"],attempt["frame"]))
            a = begin["step"]
            assert begin["planned_actions"] == [r["action"] for r in times[a:end]]
            frame_id = attempt["frame"]
            source, = [t for t in kfs if trace[t]["mapping_frame_id"]==frame_id]
            node = next(n for n in hist["final_memory"]["instances"] if n["instance_id"]==attempt["instance"])
            registration, = [o for o in hist["observations"] if o["observation_id"] in node["observation_ids"]
                             and o["frame_id"]==frame_id and o["step"]<=a]
            past_kfs = [t for t in kfs if t <= end]
            anchor = past_kfs[-1]
            strict_anchor = max(t for t in kfs if t<end)
            previous_eight=[t for t in kfs if t<end][-8:]
            steps.update(previous_eight)
            previous=None
            for prev in reversed([t for t in kfs if t<strict_anchor]):
                between=range(prev,strict_anchor)
                if not all(trace[t+1]["last_motion_failed"] is False for t in between):
                    break
                pose=(0.,0.,0.)
                for t in between:
                    pose=pose_after(pose,times[t]["action"],1.)
                travel=float(np.hypot(*pose[:2]))
                if travel>=.20:
                    previous=dict(step=prev,to_anchor_nominal_m=travel)
                    steps.add(prev)
                    break
            steps.add(strict_anchor)
            ps=(0.,0.,0.)
            for t in range(a,end):
                assert int(trace[t]["actual_action"]["id"]) == times[t]["action"]
                assert trace[t+1]["last_motion_failed"] is False
                ps=pose_after(ps,times[t]["action"],1.)
            steps.update([source,a,end,anchor])
            archive = dict(source=source,start=a,end=end,anchor=anchor,source_box=registration["bbox"],
                           source_click=registration["pixel"],displacement_m=float(np.hypot(*ps[:2])),
                           prior_keyframe=strict_anchor,earlier_translated_keyframe=previous,prior_eight_keyframes=previous_eight)
            obs = dict(arrival_step=end,source_keyframe=True,current_keyframe=end in kfs,
                       previous_keyframe_age=end-anchor,movement_actions=end-a)
            cases.append((archive,obs))
        if not cases:
            continue
        env,item,data = semantic_scene(doc)
        target_ids = set(map(int,item.target_object_ids))
        gt = {}
        try:
            f = _reset_environment(env,item,data)
            for step,t in enumerate(times):
                assert step==t["step"]
                if step in steps:
                    assert np.linalg.norm(state(env)[0]-t["position_before"]) < .002
                    native = np.asarray(f["rgb"])[...,:3]
                    image = Image.fromarray(native).resize((518,518),Image.Resampling.BICUBIC)
                    image.save(PRIVATE / f"{name}_{step}.png")
                    m = np.asarray(f["semantic"]).squeeze()
                    assert m.shape == (320,320)
                    near = set(map(int,_near_target_ids(tuple(state(env)[0]),item.target_object_ids,item.target_positions,
                             threshold=.25,pathfinder=env.sim.pathfinder,target_view_points=item.view_points)))
                    gt[step] = m.copy(), near
                if step>=max(steps):
                    break
                if t["action_role"]=="navigate":
                    f=env.step(_env_action(t["action"]))
        finally:
            env.close()
        model_inputs=[]
        relook_near_targets=set()
        for c,obs in cases:
            src,_=gt[c["source"]]
            x,y=np.clip(np.round(np.asarray(c["source_click"])*.32),0,319).astype(int)
            iid=int(src[y,x])
            source_click_valid=iid in target_ids
            xx,yy,uu,vv=np.clip(np.round(np.asarray(c["source_box"])*320/518),0,320).astype(int)
            box_overlap={i:int(np.count_nonzero(src[yy:vv,xx:uu]==i)) for i in target_ids}
            ids_in_box=[i for i,n in box_overlap.items() if n>0]
            mask,near=gt[c["end"]]
            relook_near_targets.update(near)
            obs.update(source_click_on_task_instance=source_click_valid,
                       source_box_task_pixels=sum(box_overlap.values()),
                       source_box_area=(int(uu)-int(xx))*(int(vv)-int(yy)),
                       source_task_objects_in_box=sum(n>0 for n in box_overlap.values()),
                       source_instance_pixels_visible=int(np.count_nonzero(mask==iid)) if source_click_valid else None,
                       source_instance_in_official_range=iid in near if source_click_valid else None,
                       total_visible_task_pixels=sum(int(np.count_nonzero(mask==i)) for i in target_ids),
                       task_instances_in_official_range=len(near))
            if len(ids_in_box)==1:
                goal,=ids_in_box
                obs.update(box_overlap_instance_in_range=goal in near,box_overlap_instance_pixels_visible=int(np.count_nonzero(mask==goal)))
                Image.fromarray((mask==goal).astype(np.uint8)).save(PRIVATE/f"{name}_{c['end']}_truth.png")
            model_inputs.append({k:v for k,v in c.items() if k!="source_click"})
            state_summary["completed_rechecks"].append(obs)
        save(PRIVATE/(name+"_inputs.json"),model_inputs)
        state_summary["distinct_official_in_range_targets_across_arrivals"]=len(relook_near_targets)
        save(BASE/"archive_summary.json",summary)
    save(BASE/"archive_summary.json",summary)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
