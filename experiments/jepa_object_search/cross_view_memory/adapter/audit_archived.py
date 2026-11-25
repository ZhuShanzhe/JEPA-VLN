"""Causal availability of native object proposals in the archived clock/painting runs."""
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_bytes())


def audit(project):
    pilot = project / "mapping_runtime/pilots/jepa_confirm_v18_first2"
    result = []
    for row in read(pilot / "panel.json")["rows"]:
        name = row["episode_id"] + ".json"
        accepted = read(pilot / "comparison/accepted/second" / name)
        path = Path(accepted["result"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == accepted["result_sha256"]
        assert read(pilot / "replay" / name)["source_result_sha256"] == accepted["result_sha256"]
        ep, = read(path)["episodes"]
        observations = ep["paper"]["agent_trace"]["observations"]
        trace = path.parent / "agent/action_trace.jsonl"
        actions = {a["step"]: a for a in (json.loads(s) for s in trace.read_bytes().splitlines() if s)}
        report_frames = []
        for report in ep["report_events"]:
            step = report["step"]
            action = actions[step]
            fid = action["mapping_frame_id"] if action["mapping"].get("is_keyframe") else None
            prior = [o for o in observations if o["step"] <= step and o.get("bbox") is not None]
            prior_frames = set(o["frame_id"] for o in prior if o["frame_id"] is not None)
            current = [o for o in prior if fid is not None and o["frame_id"] == fid]
            report_frames.append({
                "step": step, "current_rgb_is_new_mapper_keyframe": fid is not None,
                "recorded_current_rgb_files": len(list((path.parent / "agent/vlm_inputs").glob(
                    f"*step-{step}_call-*_decision_00-current_observation.jpg"))),
                "independent_boxes_on_current_rgb_at_or_before_report": len(current),
                "earlier_different_frames_with_box": len(prior_frames - {fid}),
                "can_form_cross_view_box_pair_at_report": bool(current and prior_frames - {fid}),
            })
        result.append({"panel_ordinal": row["ordinal"], "native_observations": len(observations),
                       "native_box_frames_in_episode": len(set(o["frame_id"] for o in observations
                       if o.get("bbox") is not None and o.get("frame_id") is not None)),
                       "reports": report_frames})
    return {"scope":"Availability check for the archived causal report frames, without surrogate boxes, later detections, simulator identity or paid inference.", "episodes":result}


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1])), indent=2))
