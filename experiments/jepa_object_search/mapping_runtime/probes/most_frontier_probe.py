"""Run with the official MOST evaluator's --agent most_frontier_probe:ProbeAgent."""
import importlib
import json
import math
import os
from pathlib import Path

import numpy as np

from agents import navigator as nav
from agents.nav_agent import NavAgent


def relative_xy(point, pose, scale):
    dx, dy = (float(point[i]) - pose[i] for i in (0, 1))
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return [(c * dx + s * dy) * scale, (-s * dx + c * dy) * scale]


def nominal_plan(path, pose, scale, reach):
    """Free-run a fresh author follower on a copied path, without simulator calls."""
    follower = nav.PathFollower(scale=scale, reach_m=reach)
    follower.set_path(list(path))
    follower.x, follower.y, follower.yaw = pose
    actions = []
    for _ in range(24):
        action, arrived = follower.next_action()
        if arrived or action is None:
            break
        actions.append(int(action))
        follower.dead_reckon(action)
    return actions


def candidate_rows(clusters, pose, scale, reach):
    rows = []
    for idx, cluster in enumerate(clusters):
        path = cluster["path"]
        row = {"id": "f%d" % idx, "branch_id": cluster.get("branch_id"),
               "path_cost_m": float(cluster["path_cost_m"]),
               "geometry_gain": int(cluster.get("geometry_gain", 0)),
               "semantic_gain": int(cluster.get("semantic_gain", 0)),
               "failure_count": int(cluster.get("failure_count", 0)),
               "utility": float(cluster["utility"]),
               "endpoint_forward_left_m": None, "path_forward_left_m": None,
               "nominal_action_ids": None}
        if pose is not None and scale:
            row["endpoint_forward_left_m"] = relative_xy(cluster["world"], pose, scale)
            row["path_forward_left_m"] = [relative_xy(p, pose, scale) for p in path]
            row["nominal_action_ids"] = nominal_plan(path, pose, scale, reach)
        rows.append(row)
    return rows


def check_scores(scores, rows):
    if not isinstance(scores, dict) or set(scores) != {r["id"] for r in rows}:
        raise ValueError("The scorer must return exactly the current frontier IDs")
    result = {}
    for key, value in scores.items():
        if isinstance(value, (bool, str, bytes)) or not math.isfinite(float(value)):
            raise ValueError("Expected a finite numeric score for " + key)
        result[key] = float(value)
    return result


def inverse_cost_demo(*, rgb, goal_text, candidates, prior_report_steps):
    """Dependency-free wiring check. Higher scores denote shorter paths."""
    return {r["id"]: 1.0 / (1.0 + r["path_cost_m"]) for r in candidates}


class ProbeAgent(NavAgent):
    def __init__(self):
        super().__init__()
        self.probe_path = Path(self.output_dir) / "frontier_probe.jsonl"
        self.probe_write_error = False
        self.scorer = None
        self.advise = os.environ.get("MOST_FRONTIER_ADVISE", "0") == "1"
        spec = os.environ.get("MOST_FRONTIER_SCORER")
        if spec:
            module, name = spec.split(":", 1)
            self.scorer = getattr(importlib.import_module(module), name)
        if self.advise and self.scorer is None:
            raise ValueError("MOST_FRONTIER_ADVISE requires MOST_FRONTIER_SCORER")

    def _write_probe(self, observation, event, **data):
        record = {"schema": 1, "event": event, "episode": str(observation.episode_id),
                  "step": int(observation.step_count),
                  "revision": int(self._frontier_revision), **data}
        try:
            with self.probe_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        except (TypeError, ValueError, OSError) as exc:
            if not getattr(self, "probe_write_error", False):
                print(f"[frontier probe] trace incomplete: {exc}", flush=True)
                self.probe_write_error = True

    def _rows(self):
        pose = self._frontier_pose or self._frontier_slam_pose
        return candidate_rows(self._last_frontier_clusters, pose, self._frontier_scale,
                              self.reach_m)

    def _build_decider_input(self, observation, local_map=False):
        state, image = super()._build_decider_input(observation, local_map=local_map)
        rows = self._rows()
        if [r["id"] for r in rows] != [r["id"] for r in state["frontiers"]]:
            raise RuntimeError("Probe and author frontier table disagree")
        scores = None
        score_error = None
        report_steps = [int(claim.step) for claim in self.instance_store.report_claims]
        if self.scorer and rows:
            # A plugin receives independent copies, never the agent, evaluator or observation.
            try:
                scores = check_scores(self.scorer(
                    rgb=np.array(observation.rgb, copy=True), goal_text=str(observation.goal_text),
                    candidates=json.loads(json.dumps(rows)), prior_report_steps=list(report_steps)), rows)
            except Exception as exc:
                score_error = type(exc).__name__ + ": " + str(exc)
            if self.advise and scores is not None:
                state = dict(state)
                state["frontiers"] = [dict(r, advisory_score_higher_is_better=scores[r["id"]])
                                      for r in state["frontiers"]]
        self._write_probe(observation, "decision_input", candidates=rows, scores=scores, score_error=score_error,
                          advice_visible=bool(self.advise and scores), prior_report_steps=report_steps,
                          frontier_stats=dict(self._frontier_stats or {}))
        return state, image

    def _activate_frontier(self, observation, cluster, source):
        # Source code assigns fN by list position in the current snapshot, never by geometry.
        idx = next((i for i, r in enumerate(self._last_frontier_clusters) if r is cluster), None)
        activated = super()._activate_frontier(observation, cluster, source)
        self._write_probe(observation, "activation", source=str(source),
                          selected_id=None if idx is None else "f%d" % idx,
                          candidates=self._rows(), activated=bool(activated))
        return activated

    def act(self, observation):
        action = super().act(observation)
        self._write_probe(observation, "returned_action", action_id=int(action),
                          frame_id=self._last_feed_info.get("frame_id"),
                          rgb_shape=list(np.shape(observation.rgb)),
                          prior_action=observation.previous_action,
                          own_report_count=int(self._reported_count),
                          pose_free=bool(self.pose_free), scale_source=str(self.scale_source))
        return action
