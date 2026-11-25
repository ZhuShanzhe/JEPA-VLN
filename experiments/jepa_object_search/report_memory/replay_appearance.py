"""Audit the appearance condition alone using the accepted V18 recordings."""
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from appearance import build_reference, match_reference


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def current_captures(folder, step):
    return sorted(folder.glob(f"*step-{step}_call-*_decision_00-current_observation.jpg"))


def run(project):
    base = project / "mapping_runtime/pilots/jepa_confirm_v18_first2"
    records = []
    for row in read(base / "panel.json")["rows"]:
        name = row["episode_id"] + ".json"
        accepted = read(base / "comparison/accepted/second" / name)
        path = Path(accepted["result"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == accepted["result_sha256"] == read(base / "replay" / name)["source_result_sha256"]
        ep, = read(path)["episodes"]
        agent_dir = path.parent / "agent"
        actions = lines(agent_dir / "action_trace.jsonl")
        events = lines(agent_dir / "confirm_events.jsonl")
        reports = {r["step"]: r for r in ep["report_events"]}
        emitted = [e for e in events if e["event"] in ("report_issued", "native_report_without_template")]
        assert set(reports) == {e["step"] for e in emitted}
        table = ep["paper"]["agent_trace"]
        observations = {o["observation_id"]: o for o in table["observations"]}
        frames = agent_dir / "vlm_inputs"
        for node in table["final_memory"]["instances"]:
            # Later observations or duplicate attachments cannot explain an earlier action.
            obs = min((observations[i] for i in node["observation_ids"]), key=lambda o: (o["step"], o["observation_id"]))
            frame = obs["frame_id"]
            captures = sorted(frames.glob(f"*tool_frame_{frame}_rgb.jpg"))
            source = "retrieved_keyframe"
            if not captures:
                step, = [a["step"] for a in actions if a["mapping_frame_id"] == frame and a["mapping"].get("is_keyframe")]
                captures = current_captures(frames, step)
                source = "current_rgb_same_keyframe_step"
            assert captures
            audited = []
            for capture in captures:
                raw = capture.read_bytes()
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                assert image.shape[0] == image.shape[1]
                ref = build_reference(raw, obs["bbox"], grid_size=(518,518), point_1000=obs["pixel"])
                crop_size = list(ref.patch.shape[:2]) if ref is not None else None
                std = round(float(ref.patch.reshape(-1,3).std(axis=0).max()), 2) if ref is not None else None
                fixed_reports = []
                for event in emitted:
                    if event["instance"] != node["instance_id"]:
                        continue
                    report_step = event["step"]
                    assert obs["step"] <= report_step
                    rgb_files = current_captures(frames, report_step)
                    assert rgb_files
                    evidence = [match_reference(ref, image_file.read_bytes()) for image_file in rgb_files]
                    assert len(set((m.matched, m.score, m.margin, m.reason) for m in evidence)) == 1
                    fixed_reports.append(dict(step=report_step, rgb_copies=len(rgb_files), new_credits_in_archived_run=len(reports[report_step]["new_targets"]),
                        appearance_passed=evidence[0].matched, score=evidence[0].score, margin=evidence[0].margin, reason=evidence[0].reason))
                audited.append(dict(ordinal=row["ordinal"], registered_step=obs["step"], rgb_source=source,
                    source_dimensions_hw=list(image.shape[:2]), source_copies=len(captures), template_created=ref is not None,
                    template_dimensions_hw=crop_size, max_channel_std=std, archived_reports=fixed_reports))
            assert all(a == audited[0] for a in audited)
            records.append(audited[0])
    assert sorted((r["ordinal"],r["registered_step"]) for r in records)==[(1,12),(14,218),(14,239)]
    return dict(scope="Prior V18 initial causal evidence and fixed archived native report images. This tests only V19 full-image template matching, not live decision or navigation outcomes.",
                mapper_bbox_grid_wh=[518,518], records=records)


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1])), indent=2, allow_nan=False))
