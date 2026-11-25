"""Untrained, fixed-caption detector diagnostic on archived report RGBs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from audit_archived import audit, read


def detect(project):
    import cv2
    import numpy as np
    import torch
    from model_worker import MatcherProcess
    from regions import crop_evidence
    torch.set_num_threads(4)
    detector = project / "b_verification/detector_run"
    os.chdir(detector)
    sys.path.insert(0, str(detector))
    from groundingdino.util.inference import load_image, load_model, predict
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py",
                       "weights/groundingdino_swint_ogc.pth", device="cpu")
    availability = audit(project)
    pilot = project / "mapping_runtime/pilots/jepa_confirm_v18_first2"
    benchmark_config = project / "frontier_scenes/most_release/evaluation/main/hm3d_config.yaml"
    assert "turn_angle: 30" in benchmark_config.read_text()
    worker = MatcherProcess(project.parent / "jepa_instance_bench_20261016/policy_env/bin/python",
                            project, project/"jepa_reid_v20/model/jepa.pt","cpu")
    results = []
    for row, previous in zip(read(pilot / "panel.json")["rows"], availability["episodes"]):
        assert row["ordinal"] == previous["panel_ordinal"]
        caption = {1: "clock.", 14: "painting."}[row["ordinal"]]
        name = row["episode_id"] + ".json"
        path = Path(read(pilot / "comparison/accepted/second" / name)["result"])
        ep, = read(path)["episodes"]
        table = ep["paper"]["agent_trace"]
        nodes = {n["instance_id"]:n for n in table["final_memory"]["instances"]}
        observations = {o["observation_id"]:o for o in table["observations"]}
        agent = path.parent / "agent"
        frames = agent / "vlm_inputs"
        actions = {a["step"]: a for a in map(json.loads,(agent/"action_trace.jsonl").read_bytes().splitlines())}
        emitted = {e["step"]:e["instance"] for e in map(json.loads,(agent/"confirm_events.jsonl").read_bytes().splitlines())
                   if e["event"] in ("report_issued","native_report_without_template")}
        for report in previous["reports"]:
            step = report["step"]
            frame, = (path.parent / "agent/vlm_inputs").glob(
                f"*step-{step}_call-*_decision_00-current_observation.jpg")
            target_rgb, image = load_image(str(frame))
            boxes, scores, labels = predict(model, image, caption, 0.35, 0.25, device="cpu")
            node = nodes[emitted[step]]
            registered = min((observations[oid] for oid in node["observation_ids"]),
                             key=lambda r: (r["step"],r["observation_id"]))
            assert registered["step"] <= step
            frame_id = registered["frame_id"]
            source_step, = [s for s,a in actions.items() if a.get("mapping_frame_id") == frame_id
                            and a.get("mapping", {}).get("is_keyframe")]
            assert source_step <= registered["step"]
            originals = sorted(frames.glob(f"*tool_frame_{frame_id}_rgb.jpg"))
            if not originals:
                originals=sorted(frames.glob(f"*step-{source_step}_call-*_decision_00-current_observation.jpg"))
            assert originals
            original=originals[0].read_bytes()
            dims = cv2.imdecode(np.frombuffer(original,np.uint8),cv2.IMREAD_COLOR).shape[:2]
            assert dims[0]==dims[1]
            source=crop_evidence(original,np.array(registered["bbox"],float)*dims[0]/518.)
            assert source is not None
            # Source and report images are captured before the action at their step.
            # Use only the stream of actually observed prior actions, never world/GT rotation.
            corrections = {"TURN_LEFT":-30, "TURN_RIGHT":30}
            locks = [actions[i]["previous_action"]["name"] for i in range(source_step+1,step+1)]
            assert all(actions[i]["actual_action"]["name"] == actions[i+1]["previous_action"]["name"]
                       for i in range(source_step,step))
            assert all(t in {"TURN_LEFT","TURN_RIGHT","MOVE_FORWARD","LOOK_UP","LOOK_DOWN","TARGET_FOUND",
                             "STOP","FOUND","FINISH","REPORT_FOUND","NOOP"} for t in locks),set(locks)
            yaw = math.remainder(sum(corrections.get(t,0) for t in locks),360)
            matches=[]
            h,w=target_rgb.shape[:2]
            jpeg=frame.read_bytes()
            for box,det_score in zip(boxes.tolist(),scores):
                cx,cy,bw,bh=box
                candidate_box=np.clip([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h],
                                      [0,0,0,0],[w,h,w,h])
                region=crop_evidence(jpeg,candidate_box)
                result=worker.compare(source,region,yaw) if region is not None else {"reason":"unusable_region"}
                matches.append({"detector_score":round(float(det_score),4),**result})
            results.append({"ordinal": row["ordinal"], "step": step, "caption": caption,
                            "first_registration_step":registered["step"],"reference_capture_step":source_step,
                            "observed_action_heading_change_deg":yaw,
                            "rgb_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                            "candidate_boxes": len(boxes), "scores": [round(float(s),4) for s in scores],
                            "phrases": list(labels),"model_pairs":matches})
    worker.close()
    return {"detector":"GroundingDINO Swin-T OGC, official pretrained local weights",
            "box_threshold":0.35,"text_threshold":0.25,"gt_boxes_used":False,
            "heading_source":"Accumulated observed actions, standard 30-degree turn; source and target poses cancel the initial azimuth.",
            "scope":"Supplementary retrospective 2D proposals/JEPA scores. No independent identity labels or live navigation claims.",
            "rows":results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = detect(args.project)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
