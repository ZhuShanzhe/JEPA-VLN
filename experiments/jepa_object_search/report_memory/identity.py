"""Conservative identity checks on MOST observations belonging to one episode."""

from itertools import product
from math import isfinite
from numbers import Integral


def _get(record, key):
    return record.get(key) if isinstance(record, dict) else getattr(record, key, None)


def _index(value):
    if isinstance(value, Integral) and not isinstance(value, bool) and value >= 0:
        return int(value)
    if isinstance(value, str) and value.isascii() and value.isdigit():
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _observation(record):
    oid = _index(_get(record, "oid"))
    if oid is None:
        oid = _index(_get(record, "observation_id"))
    handle = _get(record, "candidate_id")
    handle = handle.strip() if isinstance(handle, str) else None
    frame = _index(_get(record, "frame_id"))
    raw = _get(record, "bbox")
    bbox = None
    if raw is not None and not isinstance(raw, (str, bytes, dict)):
        try:
            x0, y0, x1, y1 = map(float, raw)
            if (
                all(map(isfinite, (x0, y0, x1, y1)))
                and 0 <= x0 < x1
                and 0 <= y0 < y1
                and isfinite((x1 - x0) * (y1 - y0))
            ):
                bbox = x0, y0, x1, y1
        except (ValueError, TypeError, OverflowError):
            pass
    return oid, handle or None, frame, bbox


def _box_overlap(a, b):
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    if intersection == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def relation(left_observations, right_observations, left_goal=None, right_goal=None):
    """Return same, distinct, or unknown for two sets of native observation records.

    Accepts native ObservationRecord objects or their as_dict() representation;
    image goals must use the one-image-per-instance convention. No coordinates
    are compared across camera frames. Ambiguity must leave the other ID eligible.
    """
    left_goal, right_goal = _index(left_goal), _index(right_goal)
    if left_goal is not None and right_goal is not None and left_goal != right_goal:
        return "distinct"
    same = False
    for left, right in product(
        map(_observation, left_observations or ()), map(_observation, right_observations or ())
    ):
        oid, handle, frame, a = left
        other_oid, other_handle, other_frame, b = right
        if frame is not None and frame == other_frame and a and b:
            overlap = _box_overlap(a, b)
            if overlap == 0:
                return "distinct"
            same |= overlap >= 0.90  # The native same-frame replay threshold.
        same |= (oid is not None and oid == other_oid) or (handle is not None and handle == other_handle)
    return "same" if same else "unknown"


class GuardedReplayMixin:
    """Recognize exact candidate transactions and unique same-frame box matches.

    A candidate with no usable box and a fresh handle goes to native review.
    Center proximity alone cannot suppress a different nearby object.
    """

    def _find_replay_observation(self, candidate_id=None, frame_id=None, pixel=None, bbox=None):
        query = {"candidate_id": candidate_id, "frame_id": frame_id, "bbox": bbox}
        _, handle, frame, box = _observation(query)
        if handle is not None:
            oid = self._candidate_to_observation.get(handle)
            if oid is not None:
                hit = self.get_observation(oid)
                # Aliases can point at observations whose initial handle differed.
                if hit is not None and relation([query], [hit]) != "distinct":
                    return hit
        if frame is None or box is None:
            return None
        matches = []
        owners = set()
        for obs in self.observations.values():
            _, _, other_frame, other_box = _observation(obs)
            if frame != other_frame or other_box is None:
                continue
            overlap = _box_overlap(box, other_box)
            if overlap < self.exact_bbox_iou:
                continue
            owner = self.instance_for_observation(obs.oid)
            owners.add(("pending", obs.oid) if owner is None else ("instance", owner.iid))
            matches.append((overlap, obs.oid, obs))
        if len(owners) != 1:
            return None
        return max(matches, key=lambda row: (row[0], -row[1]))[2]


def guarded_instance_memory():
    """Construct an empty per-episode native registry; independent of optional simulator imports."""
    from agents.instance_store import InstanceMemory

    class GuardedInstanceMemory(GuardedReplayMixin, InstanceMemory):
        pass

    return GuardedInstanceMemory()
