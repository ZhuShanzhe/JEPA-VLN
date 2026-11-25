"""Apply a bounded preference against looking at the same reported view again."""

import math

import numpy as np


# Camera distance, in metric units after alignment; this is not a room exclusion radius.
VIEW_RADIUS_M = 0.55
VIEW_HEADING_RAD = math.radians(25)
PENDING_RADIUS_M = 1.5
MAX_NEW_RAY_FRACTION = 0.25
MAX_PENALTY = 0.15


def _prefix_poses(start, scale, actions):
    x, y, yaw = start
    poses = [(x, y, yaw)]
    for action in actions:
        if action == 1:
            x += 0.25 * math.cos(yaw) / scale
            y += 0.25 * math.sin(yaw) / scale
        elif action in (2, 3):
            yaw += math.pi / 6 if action == 2 else -math.pi / 6
        else:
            return []
        poses.append((x, y, yaw))
    return poses


def _finite_rows(values, width):
    if values is None:
        return None
    out = []
    for value in values:
        row = np.asarray(value, dtype=float)
        if row.shape != (width,) or not np.isfinite(row).all():
            return None
        out.append(tuple(map(float, row)))
    return out


def _evaluate(plan, profile, start, scale, anchors, pending):
    if not profile["allowed"]:
        return 0.0, "inadmissible"
    steps, gain = profile["view_steps"], profile["new_view_fraction"]
    if not 0 < steps <= len(plan) or not 0 <= gain <= 1:
        return 0.0, "missing_view"
    if gain >= MAX_NEW_RAY_FRACTION:
        return 0.0, "novel_field_of_view"
    poses = _prefix_poses(start, scale, plan[:steps])
    if not poses:
        return 0.0, "unknown_action"
    end = poses[-1]
    repeated = any(
        math.dist(end[:2], old[:2]) * scale <= VIEW_RADIUS_M
        and abs(math.remainder(end[2] - old[2], 2 * math.pi)) <= VIEW_HEADING_RAD
        for old in anchors
    )
    if not repeated:
        return 0.0, "different_view"
    if any(math.dist(p[:2], target) * scale <= PENDING_RADIUS_M for target in pending for p in poses):
        return 0.0, "pending_instance_near_path"
    return MAX_PENALTY * (1 - gain / MAX_NEW_RAY_FRACTION), "repeated_report_view"


class MemoryChannel:
    """Wrap V18 ViewChannel. Only penalize its admitted physical observation prefixes."""

    def __init__(self, stream, host):
        self.stream = stream
        self.host = host

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def send(self, msg):
        self.plans = [list(plan) for plan in msg["plans"]]
        self.step = msg["step"]
        pose = self.host._frontier_pose
        if pose is None:
            pose = self.host._frontier_slam_pose
        self.start = tuple(pose) if pose is not None else None
        self.scale = self.host._frontier_scale
        self.anchors = _finite_rows(self.host._reported_views_for_planning(), 3)
        self.pending = (
            _finite_rows(self.host._unreported_points_for_planning(), 2) if self.anchors else None
        )
        return self.stream.send(msg)

    def recv(self):
        raw = self.stream.recv()
        rows = raw["view_profiles"]
        scores = np.asarray(raw["final_scores"], dtype=float)
        old = raw["index"]
        out = dict(raw, pre_report_memory_index=old)
        penalties = np.zeros(len(scores))
        reasons = ["no_applicable_report_memory"] * len(scores)
        valid_pose = self.start is not None and len(self.start) == 3 and np.isfinite(self.start).all()
        if (
            self.anchors
            and self.pending is not None
            and valid_pose
            and self.scale is not None
            and np.isfinite(self.scale)
            and self.scale > 0
            and raw["view_reason"] == "view_ranked"
            and len(self.plans) == len(rows) == len(scores)
        ):
            allowed = np.asarray([p["allowed"] for p in rows], dtype=bool)
            for i, (plan, row) in enumerate(zip(self.plans, rows)):
                penalties[i], reasons[i] = _evaluate(
                    plan, row, self.start, self.scale, self.anchors, self.pending
                )
            # Keep the native ranking if every feasible view has the same liability.
            if not np.any(allowed & (penalties == 0)):
                reasons = ["no_unpenalized_alternative" if p else r for p, r in zip(penalties, reasons)]
                penalties[:] = 0
            if penalties.any():
                final = scores - penalties
                gain = np.array([row["new_view_fraction"] for row in rows])
                deficit = raw["clue_deficit"]
                no_evidence = (
                    np.asarray(raw["memory_scores"])
                    - penalties
                    + np.asarray(raw["unvisited_fraction"])
                    * raw["adaptive_weight"]
                    * 0.35
                    * (1 - deficit + deficit * gain)
                )

                def best(s):
                    return int(np.argmax(np.where(allowed, s, -np.inf)))

                out.update(
                    index=best(final),
                    final_scores=final.tolist(),
                    no_exploration_index=best(final - np.asarray(raw["exploration_bonus"])),
                    no_evidence_index=best(no_evidence),
                )
        out.update(report_view_penalties=penalties.tolist(), report_view_reasons=reasons)
        self.last_choice = (self.step, out["index"], rows)
        self.stream.last_choice = self.last_choice
        return out
