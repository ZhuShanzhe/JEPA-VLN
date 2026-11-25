"""Ground a native registered region against a live RGB view; no instance identity decisions."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


WORK_WIDTH = 320
SCALES = (0.60, 0.80, 1.0, 1.25, 1.6, 2.0)


@dataclass(frozen=True, eq=False)
class AppearanceReference:
    patch: np.ndarray
    roi: tuple
    context: np.ndarray


@dataclass(frozen=True)
class AppearanceMatch:
    matched: bool
    score: float
    margin: float
    box: tuple | None
    reason: str


def _rgb(image):
    if isinstance(image, (bytes, bytearray, memoryview)):
        if not len(image):
            return None
        bgr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = np.asarray(image)
    if (
        image.ndim == 3
        and image.shape[2] in (3, 4)
        and image.dtype == np.uint8
        and min(image.shape[:2]) > 0
    ):
        return np.ascontiguousarray(image[:, :, :3])
    return None


def _working(image):
    height, width = image.shape[:2]
    return cv2.resize(
        image, (WORK_WIDTH, max(1, round(height * WORK_WIDTH / width))), interpolation=cv2.INTER_AREA
    )


def build_reference(jpeg_or_rgb, bbox_xyxy_grid, *, grid_size=None, point_1000=None):
    """Use the native SAM-derived xyxy bbox on get_frame_image's *pixel* grid.

    An optional explicit grid size is (width,height), for a resampled copy of
    the same image. The separately recorded pixel is xy normalized to 0..1000.
    A slightly inaccurate center is allowed; an isolated click is insufficient.
    """
    image = _rgb(jpeg_or_rgb)
    if image is None:
        return None
    box = np.asarray(bbox_xyxy_grid, dtype=float)
    grid = np.asarray(grid_size if grid_size is not None else image.shape[1::-1], dtype=float)
    if box.shape != (4,) or grid.shape != (2,) or not np.all(np.isfinite(grid)) or np.any(grid < 1):
        return None
    if (
        not np.all(np.isfinite(box))
        or np.any(box[:2] < 0)
        or np.any(box[2:] > grid)
        or np.any(box[2:] <= box[:2])
    ):
        return None
    lo, hi = box[:2] / grid, box[2:] / grid
    if point_1000 is not None:
        center = np.asarray(point_1000, dtype=float) / 1000
        if (
            center.shape != (2,)
            or not np.all(np.isfinite(center))
            or np.any(center < 0)
            or np.any(center > 1)
        ):
            return None
        # Small centroid/box disagreement can result from raster rounding.
        if np.any(center < lo - 0.025) or np.any(center > hi + 0.025):
            return None
    if np.prod(hi - lo) > 0.70:
        return None
    frame = _working(image)
    dims = np.array(frame.shape[1::-1])
    x0, y0 = np.floor(lo * dims).astype(int)
    x1, y1 = np.ceil(hi * dims).astype(int)
    patch = frame[y0:y1, x0:x1]
    if min(patch.shape[:2]) < 8:
        return None
    # Only pixels inside the registered object box enter this check. Structure in
    # a frame around an otherwise plain painting can suffice. A flat wall cannot.
    contrast = patch.reshape(-1, 3).std(axis=0).max()
    if contrast < 7:
        return None
    return AppearanceReference(
        patch.copy(), tuple(np.r_[lo, hi]), cv2.resize(frame, (24, 24), interpolation=cv2.INTER_AREA)
    )


def same_image_region(a, b):
    """Two boxes expressed in the same live image strongly overlap."""
    if a is None or b is None:
        return False
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite([a, b]).all():
        return False
    sizes = np.array([a[2:] - a[:2], b[2:] - b[:2]])
    if np.any(sizes <= 0) or np.any(a[:2] < 0) or np.any(b[:2] < 0):
        return False
    overlap = float(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod())
    areas = sizes.prod(axis=1)
    return overlap >= 0.80 * areas.min() and overlap >= 0.55 * areas.max()


def match_registered_view(reference, current_rgb):
    """Verify this region of a caller-identified, still-current capture.

    Image checks can reject a stale capture but cannot prove capture identity.
    Do not call on historical poses inferred from similar appearance or proximity.
    """
    live = _rgb(current_rgb)
    if reference is None or live is None:
        return AppearanceMatch(False, 0.0, 0.0, None, "missing_evidence")
    frame = _working(live)
    small = cv2.resize(frame, (24, 24), interpolation=cv2.INTER_AREA)
    difference = cv2.absdiff(reference.context, small)
    if difference.mean() > 6.0 or np.percentile(difference, 90) > 16.0:
        return AppearanceMatch(False, 0.0, 0.0, None, "different_capture_context")
    dims = np.array(frame.shape[1::-1])
    roi = np.asarray(reference.roi)
    x0, y0 = np.floor(roi[:2] * dims).astype(int)
    x1, y1 = np.ceil(roi[2:] * dims).astype(int)
    region = frame[y0:y1, x0:x1]
    if min(region.shape[:2]) < 8:
        return AppearanceMatch(False, 0.0, 0.0, None, "insufficient_registered_area")
    patch = cv2.resize(reference.patch, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    similarity = float(cv2.matchTemplate(region, patch, cv2.TM_CCOEFF_NORMED)[0, 0])
    error = float(cv2.absdiff(region, patch).mean())
    ok = similarity >= 0.88 and error <= 15.0
    return AppearanceMatch(
        ok,
        round(similarity, 4),
        0.0,
        tuple(map(int, (x0, y0, x1, y1))) if ok else None,
        "same_capture_registered_box" if ok else "different_registered_region",
    )


def _same_region(box, other):
    x0, y0, x1, y1 = box
    u0, v0, u1, v1 = other
    intersection = max(0, min(x1, u1) - max(x0, u0)) * max(0, min(y1, v1) - max(y0, v0))
    smaller = min((x1 - x0) * (y1 - y0), (u1 - u0) * (v1 - v0))
    return intersection > 0.45 * smaller


def match_reference(reference, current_rgb):
    """Return evidence and a localized box in 320-wide live image coordinates.

    Search multiple scales. Compare the strongest *spatially distinct* runner-up
    across all scales, so observing two copies cannot confirm which copy is ours.
    """
    live = _rgb(current_rgb)
    if reference is None or live is None:
        return AppearanceMatch(False, 0.0, 0.0, None, "missing_evidence")
    live = _working(live)
    template = reference.patch
    candidates = []
    for scale in SCALES:
        h, w = [round(k * scale) for k in template.shape[:2]]
        if min(h, w) < 7 or h > live.shape[0] or w > live.shape[1]:
            continue
        patch = cv2.resize(
            template, (w, h), interpolation=cv2.INTER_LINEAR if scale >= 1 else cv2.INTER_AREA
        )
        if patch.reshape(-1, 3).std(axis=0).max() < 5:
            continue
        scores = cv2.matchTemplate(live, patch, cv2.TM_CCOEFF_NORMED)
        for _ in range(4):
            _, corr, _, (x, y) = cv2.minMaxLoc(scores)
            if corr < 0.60:
                break
            region = live[y : y + h, x : x + w]
            error = float(cv2.absdiff(region, patch).mean()) / 255
            score = corr - 0.6 * error
            candidates.append((score, (x, y, x + w, y + h)))
            scores[max(0, y - h // 2) : y + h // 2 + 1, max(0, x - w // 2) : x + w // 2 + 1] = -1.0
    if not candidates:
        return AppearanceMatch(False, 0.0, 0.0, None, "different_appearance")
    best, box = max(candidates)
    other = max((s for s, b in candidates if not _same_region(box, b)), default=0.0)
    margin = best - other
    if best < 0.80:
        reason = "different_appearance"
    elif margin < 0.07:
        reason = "multiple_plausible_locations"
    else:
        reason = "localized_appearance"
    return AppearanceMatch(
        reason == "localized_appearance", round(best, 4), round(margin, 4), box, reason
    )
