"""Record the frozen review decisions without running a second detector or scorer."""
import ast
import inspect
import json
from pathlib import Path
import sys
import textwrap

from multiview_agent import MultiviewFrontier


def gate_lines(method):
    lines, start = inspect.getsourcelines(method)
    source = textwrap.dedent("".join(lines))
    tree = ast.parse(source)
    markers = (
        ("obs is None", "observation_or_scale_unavailable_or_already_assigned"),
        ("len(neighbors)", "not_exactly_one_neighbor"),
        ("not np.isfinite(dist)", "neighbor_outside_review"),
        ("not records", "missing_reference_or_known_relation"),
        ("self._distinct_at", "different_objects_supported"),
        ("ref.frame_id is None", "invalid_reference_frame_or_order"),
        ("left is None", "crop_or_calibrated_geometry_unavailable"),
        ("np.linalg.norm", "camera_distance_or_angle_outside_gate"),
        ("not out[", "below_threshold"),
    )
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            condition = ast.get_source_segment(source, node.test) or ""
            for child in node.body:
                if isinstance(child, ast.Return) and isinstance(child.value, ast.Constant) and child.value.value is None:
                    result[start + child.lineno - 1] = next(
                        (name for mark, name in markers if mark in condition), "other_gate")
    return result


_POSE = MultiviewFrontier._pose_hint
_GATES = gate_lines(_POSE)


class TraceFrontier(MultiviewFrontier):
    def _audit(self, event, **counts):
        try:
            record = dict(step=int(self.now), event=event, **counts)
            with (Path(self.output_dir) / "multiview_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        except (AttributeError, OSError, TypeError, ValueError):
            pass

    def _boxes(self, key, rgb, category):
        cached = (key, category) in self._local_boxes
        rows = super()._boxes(key, rgb, category)
        self._audit("boxes", source=str(key[0]), frame=int(key[1]), cached=cached,
                    category_present=bool(category), worker_failed=self._detector_failed, count=len(rows))
        return rows

    def _add_local_candidates(self, fid, category):
        result = super()._add_local_candidates(fid, category)
        self._audit("local_registration", frame=int(fid), added=len((result or {}).get("proposals", [])))
        return result

    def _distinct_at(self, left_records, right_records):
        distinct = super()._distinct_at(left_records, right_records)
        self._audit("geometric_guard", source=sys._getframe(1).f_code.co_name, distinct=bool(distinct))
        return distinct

    def _dup_review_payload(self):
        rows, images, omitted = super()._dup_review_payload()
        self._audit("pending_review", rows=len(rows), omitted=omitted,
                    hints=sum("pose_jepa_review" in row for row in rows))
        return rows, images, omitted

    def _pose_hint(self, row, weights):
        # Each branch of the original function runs once. Observe only its
        # return frame, without repeating map queries or crop preparation.
        existing_trace = sys.gettrace()
        if existing_trace is not None:
            self._audit("pose_trace_occupied")
            return super()._pose_hint(row, weights)
        record = {}

        def at_return(frame, event, value):
            if event == "return":
                details = frame.f_locals
                record["gate"] = "hint_emitted" if value is not None else _GATES.get(frame.f_lineno, "unclassified")
                prediction = details.get("out")
                if isinstance(prediction, dict) and "score" in prediction:
                    record.update(scored=True, score=float(prediction["score"]), threshold=float(prediction["threshold"]))
                else:
                    record["scored"] = False
            return at_return

        def only_pose(frame, event, _value):
            if event == "call" and frame.f_code is _POSE.__code__:
                frame.f_trace_lines = False
                return at_return
            return None

        sys.settrace(only_pose)
        try:
            result = super()._pose_hint(row, weights)
        finally:
            sys.settrace(existing_trace)
            self._audit("pose_review", observation=row.get("observation_id"), **record)
        return result
