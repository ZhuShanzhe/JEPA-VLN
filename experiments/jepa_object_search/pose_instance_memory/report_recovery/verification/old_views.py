"""Run the current-view evidence check on four predeclared points of the V22 paths."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def main(project, destination):
    sys.path[:0] = [str(project / folder) for folder in
        ("jepa_reid_v23/report", "jepa_report_v19", "jepa_reid_v21/detector", "jepa_exec_v14",
         "jepa_completion_v17/physical")]
    from appearance import build_reference, match_reference, match_registered_view
    from check_routes import semantic_scene
    from decision.vlm import VLMDecisionClient
    from evaluation.main.evaluator import _env_action, _reset_environment
    from proposal_worker import ProposalWorker
    from report_evidence import eligible_boxes, current_support
    from verify_habitat import deny_network, state
    sys.addaudithook(deny_network)
    runtime = project / "mapping_runtime"
    pilot = runtime / "pilots/jepa_reid_v22_first2"
    rows = read(pilot / "panel.json")["rows"]
    worker = None
    results = []
    try:
        for row in rows:
            clock = row["ordinal"] == 1
            steps = (80, 116) if clock else (78, 211)
            eid = row["episode_id"]
            if clock:
                started = min((pilot / "comparison/attempts/second" / eid).glob("*/started.json"),
                              key=lambda f: f.stat().st_mtime)
                folder = runtime / "runs" / started.parent.name / "agent"
                ref = read(Path(row["ref"]))
            else:
                ref = read(pilot / "comparison/accepted/second" / (eid + ".json"))
                folder = Path(ref["result"]).parent / "agent"
            source = Path(ref["result"])
            assert hashlib.sha256(source.read_bytes()).hexdigest() == ref["result_sha256"]
            doc = read(source)
            actions = lines(folder / "action_trace.jsonl")
            decisions = lines(folder / "decision_trace.jsonl")
            events = lines(folder / "confirm_events.jsonl")
            if not clock:
                history = doc["episodes"][0]["paper"]["agent_trace"]
                observations = {o["observation_id"]: o for o in history["observations"]}
                directory, = sorted(runtime.glob("runs/jepa_v22_reid_first2_mapping_*/mapping/diagnostics/"+eid+"_frames"))
            env, item, data = semantic_scene(doc)
            try:
                frame = _reset_environment(env, item, data)
                for i in range(max(steps) + 1):
                    assert actions[i]["step"] == i
                    if not clock:
                        native = doc["episodes"][0]["paper"]["steps"][i]
                        assert np.linalg.norm(state(env)[0] - native["position_before"]) < .002
                    if i in steps:
                        rgb = np.ascontiguousarray(frame["rgb"][:, :, :3])
                        copies = sorted((folder / "vlm_inputs").glob(f"*_step-{i}_call-*_decision_00-current_observation.jpg"))
                        assert all(f.read_bytes() == VLMDecisionClient.encode_rgb(frame["rgb"]) for f in copies)
                        local = [d["world_state"]["local_rgb_proposals"] for d in decisions if d["step"] == i]
                        source_kind = "archived_live_detections"
                        if local:
                            assert all(p["boxes"] == local[0]["boxes"] for p in local)
                            assert [local[0]["height"],local[0]["width"]] == list(rgb.shape[:2])
                            boxes = [{"bbox":b} for b in local[0]["boxes"]]
                        else:
                            if worker is None:
                                worker = ProposalWorker(project/"b_verification/env/bin/python", project, "cpu")
                            label = next(d["world_state"]["local_rgb_proposals"]["category"] for d in decisions if d["step"] < i)
                            boxes = worker.detect(rgb, label)
                            source_kind = "frozen_detector_on_replayed_live_rgb"
                        valid = eligible_boxes(boxes,rgb.shape)
                        sample = dict(task="clock" if clock else "wall_art", step=i, decision_rgb_verified=bool(copies),
                                      detections=len(boxes), eligible_box_count=len(valid), box_source=source_kind)
                        if clock:
                            # Interrupted attempts have no terminal SAM registry. Never substitute oracle boxes.
                            sample["current_instance_support"] = None
                        else:
                            dispatch, = [e for e in events if e["step"] == i and e["event"] == "report_dispatched"]
                            node, = [n for n in history["final_memory"]["instances"] if n["instance_id"] == dispatch["instance"]]
                            refs = [observations[k] for k in node["observation_ids"] if observations[k]["step"] <= i]
                            assert len(refs) == 1
                            rec = refs[0]
                            raw = (directory / ("rgb_%06d.jpg" % rec["frame_id"])).read_bytes()
                            template = build_reference(raw, rec["bbox"], grid_size=(518,518), point_1000=rec["pixel"])
                            assert template is not None
                            same_capture = actions[i]["mapping"].get("is_keyframe") and actions[i]["mapping_frame_id"] == rec["frame_id"]
                            old_match = (match_registered_view if same_capture else match_reference)(template,rgb)
                            box, reason = current_support(template,old_match,rgb,valid)
                            sample.update(current_instance_support=box is not None, evidence_reason=reason)
                        results.append(sample)
                    if i < max(steps):
                        if clock or native["action_role"] == "navigate":
                            frame = env.step(_env_action(actions[i]["actual_action"]["id"]))
            finally:
                env.close()
    finally:
        if worker is not None:
            worker.close()
    destination.write_text(json.dumps(dict(paid_calls=0,samples=results),indent=2)+"\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
