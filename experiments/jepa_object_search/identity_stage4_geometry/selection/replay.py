"""Replay the unmodified MOST keyframe selector on two frozen episode streams."""
import argparse
import ast
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input", choices=("archive", "renderer"), required=True)
    args = parser.parse_args()
    p = args.project.resolve()
    most = p/"private_sources/MOST-Nav"
    for directory in [most, most/"VGGT-SLAM", p/"jepa_exec_v14", p/"jepa_completion_v17/physical"]:
        sys.path.insert(0, str(directory))
    from mapping.keyframes import AdaptiveKeyframeSelector
    # The upstream module imports torch for its command-line demo. The actual
    # CPU tracker uses cv2 and numpy; load that class verbatim in Habitat's env.
    tracker_path = most/"VGGT-SLAM/vggt_slam/frame_overlap.py"
    tracker_node, = [n for n in ast.parse(tracker_path.read_bytes()).body
                     if isinstance(n, ast.ClassDef) and n.name == "FrameTracker"]
    scope = {"cv2":cv2, "np":np}
    exec(compile(ast.Module(body=[tracker_node], type_ignores=[]), str(tracker_path), "exec"),scope)
    FrameTracker = scope["FrameTracker"]
    cv2.setNumThreads(2)
    if args.input == "renderer":
        from check_routes import semantic_scene
        from verify_habitat import deny_network, state
        from evaluation.main.evaluator import _env_action, _reset_environment
        sys.addaudithook(deny_network)
    pilot = p/"mapping_runtime/pilots/jepa_reid_v24_first2"
    settings = {"original_replay": (40,3), "half_flow": (20,3), "shorter_interval": (40,2)}
    result = {"input": args.input, "settings": settings,
              "scope": "fixed trajectories, selection/geometry availability are distinct",
              "selector_sha256": sha(most/"mapping/keyframes.py"),
              "tracker_sha256": sha(most/"VGGT-SLAM/vggt_slam/frame_overlap.py"),
              "opencv_version": cv2.__version__, "episodes": {}}
    for item, label in zip(read(pilot/"panel.json")["rows"][:2], ["clock","painting"]):
        ticket = read(pilot/"comparison/accepted/second"/(item["episode_id"]+".json"))
        path = Path(ticket["result"])
        assert sha(path) == ticket["result_sha256"]
        doc = read(path)
        episode, = doc["episodes"]
        actions = episode["paper"]["steps"]
        trace = rows(path.parent/"agent/action_trace.jsonl")
        assert len(actions) == len(trace) == 300
        event_path = path.parent/"agent/confirm_events.jsonl"
        events = {r["step"] for r in (rows(event_path) if event_path.exists() else [])
                  if r.get("event") == "confirm_view_recheck"}
        diag = p/"mapping_runtime/runs"/ticket["evidence"]["server_run"]/"mapping/diagnostics"
        frame_dir = diag/(item["episode_id"]+"_frames")
        manifest = {r["frame_id"]: r for r in rows(diag/(item["episode_id"]+"_frame_captions.jsonl"))
                    if r.get("event") == "frame_saved"}
        selectors = {k: AdaptiveKeyframeSelector(FrameTracker(), *settings[k]) for k in settings}
        counts = {k: Counter() for k in ["logged", *selectors, "all", "event_only_extra"]}
        hits = {k: [] for k in counts}
        agreements = {k: 0 for k in selectors}
        env = None
        image_bytes = 0
        if args.input == "renderer":
            env, task, dataset = semantic_scene(doc)
            obs = _reset_environment(env, task, dataset)
        try:
            for i, (row, action) in enumerate(zip(trace, actions)):
                assert row["step"] == i
                # Non-keyframes have no public mapping_frame_id in some exports.
                fid = i + 1
                entry = manifest[fid]
                reference = row["mapping"]["is_keyframe"]
                assert reference == entry["is_keyframe"]
                if row["mapping_frame_id"] is not None and reference:
                    assert row["mapping_frame_id"] == fid
                jpeg = frame_dir/entry["image"]
                image_bytes += jpeg.stat().st_size
                if env is None:
                    bgr = cv2.imread(str(jpeg))
                else:
                    assert np.linalg.norm(state(env)[0] - action["position_before"]) < .002
                    bgr = cv2.cvtColor(np.asarray(obs["rgb"])[...,:3], cv2.COLOR_RGB2BGR)
                assert bgr.shape == (1280,1280,3)
                decisions = {"logged": (reference,entry["keyframe_reason"]), "all": (True,"fixed"),
                             "event_only_extra": (reference or i in events, "logged_or_recheck")}
                for key, selector in selectors.items():
                    decisions[key] = selector.select(bgr)
                    agreements[key] += int(bool(decisions[key][0]) == reference)
                for key, (selected, reason) in decisions.items():
                    if selected:
                        counts[key][reason] += 1
                        if i in events:
                            hits[key].append(i)
                if env is not None and action["action_role"] == "navigate" and i + 1 < len(trace):
                    obs = env.step(_env_action(action["action"]))
        finally:
            if env is not None:
                env.close()
        result["episodes"][label] = {"steps": len(trace), "historical_rechecks": sorted(events),
            "selected": {k: sum(v.values()) for k,v in counts.items()},
            "recheck_selected":hits, "agreement_with_logged":agreements,
            "archive_jpeg_bytes":image_bytes, "reason_counts":counts}
        print(label, result["episodes"][label]["selected"], agreements, hits,flush=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
