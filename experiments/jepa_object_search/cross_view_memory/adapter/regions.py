"""Extract independent mapper object regions in the training crop convention."""
import math

import cv2
import numpy as np


def crop_evidence(jpeg, bbox):
    if not jpeg or bbox is None:
        return None
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    box = np.asarray(bbox, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        return None
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        return None
    dx, dy = x1 - x0, y1 - y0
    if min(dx, dy) < 8 or dx * dy > 0.7 * w * h:
        return None

    def expanded(pad):
        left, top = max(0, math.floor(x0 - pad * dx)), max(0, math.floor(y0 - pad * dy))
        right, bottom = min(w, math.ceil(x1 + pad * dx)), min(h, math.ceil(y1 + pad * dy))
        return np.ascontiguousarray(rgb[top:bottom, left:right]), (left, top)

    obj, _ = expanded(0.15)
    context, (cx, cy) = expanded(0.80)
    mask_box = (math.floor(x0) - cx, math.floor(y0) - cy, math.ceil(x1) - cx, math.ceil(y1) - cy)
    return obj, context, mask_box


def heading_change_degrees(pose1, pose2, align):
    if pose1 is None or pose2 is None or align is None:
        return None
    align = np.asarray(align, dtype=float)
    if align.shape != (3, 3) or not np.isfinite(align).all():
        return None
    angles = []
    for pose in (pose1, pose2):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            return None
        forward = align @ pose[:3, 2]  # Positive optical Z in the mapper.
        if np.linalg.norm(forward[:2]) < 0.2:
            return None
        angles.append(math.atan2(forward[1], forward[0]))
    # MOST aligned XY increases counterclockwise (left turn). Habitat atan2(z,x)
    # increases clockwise viewed from above. A constant global azimuth cancels.
    return math.degrees(math.remainder(angles[0] - angles[1], 2 * math.pi))
