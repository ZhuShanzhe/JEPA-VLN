"""Resolve the native registration, viewpoint-return and report sequence for an accepted pilot."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def load(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def archived_texture(ep, folder, actions, observations):
    import cv2
    import numpy as np
    from view_memory import evidence_patch
    frames = {frame for _,_,frame in observations if frame is not None}
    results = []
    for obs in ep["paper"]["agent_trace"]["observations"]:
        fid, point = obs.get("frame_id"), obs.get("pixel")
        if fid not in frames or point is None:
            continue
        steps = [a["step"] for a in actions if a["mapping_frame_id"] == fid and a["mapping"].get("is_keyframe")]
        views = folder / "vlm_inputs"
        by_kind = {"retrieved_keyframe_rgb":sorted(views.glob(f"*tool_frame_{fid}_rgb.jpg"))}
        if len(steps) == 1:
            by_kind["same_step_current_rgb"] = sorted(views.glob(f"*step-{steps[0]}_call-*_decision_00-current_observation.jpg"))
        packs = []
        for kind, files in by_kind.items():
            if not files:
                continue
            values = []
            codes = []
            for file in files:
                data = file.read_bytes()
                gray = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_GRAYSCALE)
                gray = cv2.resize(gray,(320,320),interpolation=cv2.INTER_AREA)
                x,y = np.rint(np.asarray(point) * 319 / 1000).astype(int)
                assert 0 <= x < 320 and 0 <= y < 320
                crop = gray[max(0,y-14):y+15,max(0,x-14):x+15]
                values.append(round(float(crop.std()),3))
                codes.append(evidence_patch(data,point) is not None)
            packs.append(dict(source=kind, copies=len(files), min_std=min(values), max_std=max(values),
                              all_passed=all(codes), all_failed=not any(codes)))
        results.append(dict(observation_creation_step=obs.get("step"),unique_keyframe_action_count=len(steps),
                            valid_recorded_center=all(0<=v<=1000 for v in point),
                            texture_std_required=9., archival_image_checks=packs))
    return results


def summarize(ref, replay, treatment):
    receipt = load(ref)
    source = Path(receipt["result"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == receipt["result_sha256"] == load(replay)["source_result_sha256"]
    ep, = load(source)["episodes"]
    folder = source.parent / "agent"
    actions = lines(folder / "action_trace.jsonl")
    decisions = lines(folder / "decision_trace.jsonl")
    reports = {r["step"]: r for r in ep["report_events"]}
    assert sorted(reports) == [r["step"] for r in actions if r["actual_action"]["name"] == "TARGET_FOUND"]
    tools = Counter()
    commit_totals = Counter()
    commit_steps = []
    registered = set()
    registration_observations = []
    resolutions = Counter()
    geometries = {}
    noted_frames = set()
    decision_distances = {}
    for d in decisions:
        world = d.get("world_state") or {}
        geometries[d["step"]] = {"event":d["event"],"action":(d.get("output") or {}).get("action"),
            "scale_available":(d.get("agent_snapshot",{}).get("metric_transform") or {}).get("scale") is not None}
        noted_frames.update(x["frame_id"] for x in (world.get("relevant_frames") or []) if isinstance(x,dict) and x.get("frame_id") is not None)
        for n in world.get("instances") or []:
            if n.get("dist_m") is not None:
                decision_distances[d["step"], n["id"]] = n["dist_m"]
        for t in d["tools"]:
            name = t["request"]["name"]
            tools[name] += 1
            if name == "resolve_duplicate" and t["response"].get("ok"):
                resolutions[str(t["request"].get("decision"))] += 1
            if name != "commit_candidates":
                continue
            outcome = t["response"].get("result") or {}
            if not isinstance(outcome, dict):
                continue
            items = outcome.get("instances") or []
            commit_steps.append({"step": d["step"], "accepted": len(items)})
            commit_totals["accepted_observations"] += len(items)
            for k in ("rejected","uncertain","geometry_rejections"):
                commit_totals[k] += len(outcome.get(k) or [])
            registered.update(i["instance_id"] for i in items if i.get("instance_id") is not None)
            registration_observations.extend((d["step"], i["instance_id"], i.get("frame_id")) for i in items if i.get("instance_id") is not None)
    path = folder / "confirm_events.jsonl"
    events = lines(path) if treatment and path.exists() else []
    saved = {}
    return_state = {}
    matched = {}
    deferred = []
    return_history = []
    issued = {}
    for e in events:
        iid = e["instance"]
        name = e["event"]
        step = e["step"]
        if name == "view_saved":
            saved.setdefault(iid, []).append((step, e["frame"]))
        elif name == "return_started":
            state = dict(start_step=step, arrived_step=None, end_step=None, end_reason=None)
            return_history.append(state)
            return_state[iid] = state
        elif name == "return_arrived" and iid in return_state:
            state = return_state[iid]
            if state["arrived_step"] is None:
                state["arrived_step"] = step
        elif name == "return_cancelled" and iid in return_state:
            state = return_state.pop(iid)
            state.update(end_step=step, end_reason=e["reason"])
        elif name == "report_deferred":
            deferred.append(dict(step=step, best_current_patch_score=round(e["score"],3)))
        elif name == "reidentified":
            source_steps = [s for s, frame in saved.get(iid, []) if frame == e["frame"]]
            assert source_steps and min(source_steps) <= step
            matched[(step,iid)] = dict(first_saved_step=min(source_steps),
                current_to_saved_camera_m=e["view_distance_m"], patch_score=round(e["score"],3),
                peak_margin=round(e["gap"],3), scheduled_return_active=iid in return_state,
                recorded_arrival_before_report=iid in return_state and return_state[iid]["arrived_step"] is not None)
        elif name == "report_issued":
            assert step not in issued and (step,iid) in matched
            issued[step] = dict(source="current_rgb_and_native_vlm", **matched[(step,iid)])
            issued[step]["decision_snapshot_object_distance_m"] = decision_distances.get((step,iid))
        elif name == "native_report_without_template" and step in reports:
            issued[step] = dict(source="native_without_template")
    if treatment:
        assert set(issued) == set(reports)
    physical_reports = []
    for step, r in reports.items():
        item = dict(step=step, credited_new_instances=len(r["new_targets"]), in_official_range=len(r["near_targets"]))
        item.update(issued.get(step, {"source":"native_baseline"}))
        physical_reports.append(item)
    report_steps=list(reports)
    active_end=min(report_steps) if report_steps else max((a["step"] for a in actions),default=-1)
    first_save=min((s for views in saved.values() for s,_ in views),default=active_end+1)
    pre_report_decisions = [{"step":s,"action":g["action"],"scale_available":g["scale_available"]}
        for s,g in sorted(geometries.items()) if first_save<=s<active_end and g["event"] not in
        ("arrival","adjustment","nav_failed","finish_check") and g["action"] in ("GOTO_INSTANCE","GOTO_FRONTIER","EXPLORE")]
    return dict(official_reports=physical_reports, tools=dict(tools), commit_totals=dict(commit_totals),
                resolved_duplicate_tool_decisions=dict(resolutions),
                eligible_decision_kinds_after_first_save_before_first_emitted_report=pre_report_decisions,
                accepted_distinct_canonical_ids=len(registered), commit_steps=commit_steps,
                noted_relevant_frames_in_decision_history=len(noted_frames),
                accepted_observation_frames=len({f for _,_,f in registration_observations if f is not None}),
                registration_decision_current_object_distances_m=[decision_distances.get((s,i)) for s,i,f in registration_observations],
                registered_crop_checks=archived_texture(ep,folder,actions,registration_observations) if treatment else [],
                accepted_without_saved_templates=len(registered - set(saved)) if treatment else None,
                events=dict(Counter(e["event"] for e in events)), unique_saved_instances=len(saved),
                unique_saved_frames=len({frame for views in saved.values() for _,frame in views}),
                first_save_steps=sorted(min(s for s,_ in views) for views in saved.values()),
                return_attempts=return_history, deferred_reports=deferred)


def main(project, output):
    sys.path.insert(0,str(project / "jepa_confirm_v18/agent"))
    pilot = project / "mapping_runtime/pilots/jepa_confirm_v18_first2"
    result = []
    for row in load(pilot / "panel.json")["rows"]:
        name = row["episode_id"] + ".json"
        result.append(dict(ordinal=row["ordinal"], native=summarize(Path(row["ref"]),
            project / "mapping_runtime/analysis/visibility/native_20261115/verified" / name, False),
            treatment=summarize(pilot / "comparison/accepted/second" / name, pilot / "replay" / name, True)))
    output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
