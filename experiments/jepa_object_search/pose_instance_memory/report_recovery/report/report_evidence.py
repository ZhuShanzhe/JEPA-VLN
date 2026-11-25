"""Ground a registered patch in a fresh, class-conditioned live detection."""
import cv2
import numpy as np

from appearance import WORK_WIDTH


def eligible_boxes(rows, rgb_shape):
    h, w = rgb_shape[:2]
    boxes = []
    for row in rows:
        b = np.asarray(row.get("bbox", ()), dtype=float)
        if b.shape != (4,) or not np.isfinite(b).all():
            continue
        if (b[:2] < 0).any() or (b[2:] > [w, h]).any():
            continue
        extent = b[2:] - b[:2]
        # Same maximum fraction as the registered appearance reference.
        if min(extent) <= 0 or np.prod(extent) > .70*w*h:
            continue
        scale = WORK_WIDTH/w
        box = tuple((b*scale).round().astype(int))
        if min(box[2]-box[0], box[3]-box[1]) >= 7:
            boxes.append(box)
    return boxes


def overlap(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    union = np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection
    return intersection / max(1, union)


def local_correspondence(reference, candidate):
    """Verify a spatially distributed, two-way SIFT match inside both object crops."""
    if min(*reference.shape[:2], *candidate.shape[:2]) < 12 or not hasattr(cv2, "SIFT_create"):
        return False
    sift = cv2.SIFT_create(nfeatures=160, contrastThreshold=.025)
    def describe(im):
        # Scale both crops to a common long side, preserving their aspect ratios.
        scale = 128 / max(im.shape[:2])
        im = cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
        keys, desc = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY),None)
        return np.array([k.pt for k in keys], np.float32), desc, np.array(im.shape[1::-1])
    a, da, size_a = describe(reference)
    b, db, size_b = describe(candidate)
    if min(len(a), len(b)) < 6:
        return False
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def good(left,right):
        return {m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(left,right,k=2)
                if m.distance < .75*n.distance}
    ab, ba = good(da, db), good(db, da)
    pairs = [(i,j) for i,j in ab.items() if ba.get(j)==i]
    if len(pairs)<6:
        return False
    left, right = a[[i for i,j in pairs]], b[[j for i,j in pairs]]
    matrix, inlier = cv2.estimateAffinePartial2D(left, right, method=cv2.RANSAC, ransacReprojThreshold=3.)
    if matrix is None or inlier is None:
        return False
    mask=inlier.ravel().astype(bool)
    if mask.sum()<6 or mask.mean()<.67:
        return False
    for points, size in ((left[mask],size_a),(right[mask],size_b)):
        if len(np.unique(np.rint(points), axis=0)) < 6 or min(np.ptp(points,axis=0)/size)<.18:
            return False
    return True


def current_support(reference, appearance_match, rgb, boxes):
    if not boxes:
        return None, "no_local_target_box"
    # A segmented band running between opposite image boundaries has no
    # bounded object extent on that axis. Matching its scene texture in a
    # fresh box cannot establish object identity. Other references for the
    # same node can still pass independently.
    x,y,u,v=reference.roi
    if (x<=.01 and u>=.99) or (y<=.01 and v>=.99):
        return None,"reference_spans_opposite_borders"
    if appearance_match.matched and appearance_match.box is not None:
        agreeing = [b for b in boxes if overlap(appearance_match.box,b)>=.30]
        if len(agreeing)==1:
            return agreeing[0], "registered_patch_and_current_box"
        if len(agreeing)>1:
            return None, "ambiguous_live_boxes"
    h,w = rgb.shape[:2]
    live=cv2.resize(np.ascontiguousarray(rgb[:,:,:3]),(WORK_WIDTH,round(h*WORK_WIDTH/w)),interpolation=cv2.INTER_AREA)
    agreeing=[]
    for box in boxes:
        x,y,u,v=box
        if local_correspondence(reference.patch,live[y:v,x:u]):
            agreeing.append(box)
    if len(agreeing)==1:
        return agreeing[0], "registered_local_features_and_current_box"
    return None, "ambiguous_live_boxes" if agreeing else "no_current_instance_match"
