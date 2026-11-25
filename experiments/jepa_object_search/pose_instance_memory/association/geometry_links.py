"""Conservative one-to-one links between two nearby, calibrated camera views."""
from itertools import combinations
import math
import numpy as np


def area(box):
    return np.prod(np.maximum(box[2:] - box[:2], 0.))


def overlap(a, b):
    intersection = np.prod(np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0.))
    return intersection / max(area(a) + area(b) - intersection, 1.e-8)


def apart(a, b):
    # A gap in the image, not two overlapping or nested rectangles.
    if min(area(a), area(b)) < 64.:
        return False
    gap = np.maximum(a[:2] - b[2:], b[:2] - a[2:])
    return bool(np.any(gap >= 2.))


def surface_point(depth, box, K, c2w):
    """A valid RGB-D pixel near the median axial depth of the inner box."""
    K, pose = np.asarray(K, float), np.asarray(c2w, float)
    h, w = depth.shape
    a, b = np.array(box[:2]), np.array(box[2:])
    margin = .15 * (b-a)
    x0,y0 = np.maximum(np.ceil(a + margin).astype(int), 0)
    x1,y1 = np.minimum(np.floor(b - margin).astype(int), [w,h])
    if x1-x0 < 2 or y1-y0 < 2:
        return None
    tile = depth[y0:y1,x0:x1]
    ok = np.isfinite(tile) & (tile > .05) & (tile < 30.)
    if ok.sum() < 4:
        return None
    med = np.median(tile[ok])
    v,u = np.where(ok & (np.abs(tile-med) <= max(.04,med*.025)))
    if not len(u):
        return None
    j = np.argmin((u-(x1-x0-1)/2)**2 + (v-(y1-y0-1)/2)**2)
    x, y = int(u[j]+x0), int(v[j]+y0)
    ray = np.linalg.solve(K, [x+.5,y+.5,1.])
    return (pose[:3,:3] @ (ray * (depth[y,x]/ray[2])) + pose[:3,3]).tolist()


def projected_offset(point, pose, K, box):
    v = pose[:3,:3].T @ (point - pose[:3,3])
    if v[2] <= .05:
        return None
    uv = K @ v
    xy = uv[:2]/uv[2]
    size = np.maximum(box[2:]-box[:2], 4.)
    shift = (xy - (box[:2]+box[2:])*.5) / size
    if np.max(np.abs(shift)) > .59:
        return None
    return float(np.linalg.norm(shift))


def match_views(left, right):
    """No instance IDs, training labels or category-specific rules are consumed."""
    boxes = [np.asarray(f["boxes"],float).reshape(-1,4) for f in (left,right)]
    sep = [[(i,j) for i,j in combinations(range(len(bs)),2) if apart(bs[i],bs[j])] for bs in boxes]
    out = dict(matches=[], distinct=[], same_frame_distinct_left=sep[0], same_frame_distinct_right=sep[1])
    if left["frame_id"] == right["frame_id"] or any(f.get("c2w") is None or f.get("K") is None for f in (left,right)):
        return out
    poses = [np.asarray(f["c2w"],float) for f in (left,right)]
    ks = [np.asarray(f["K"],float) for f in (left,right)]
    if any(p.shape != (4,4) or not np.isfinite(p).all() or
           not np.allclose(p[:3,:3].T @ p[:3,:3],np.eye(3),atol=.003) or
           np.linalg.det(p[:3,:3]) < .99 for p in poses):
        return out
    if any(k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0 for k in ks):
        return out
    angle = math.degrees(math.acos(np.clip(poses[0][:3,2] @ poses[1][:3,2],-1.,1.)))
    displacement = np.linalg.norm(poses[0][:3,3]-poses[1][:3,3])
    if angle > 45. or displacement > 1.25 or (angle < 1. and displacement < .03):
        return out
    cost = np.full((len(boxes[0]),len(boxes[1])),np.inf)
    for i,lp in enumerate(left["points"]):
        for j,rp in enumerate(right["points"]):
            if lp is None or rp is None:
                continue
            a,b = np.asarray(lp,float),np.asarray(rp,float)
            if not (a.shape==b.shape==(3,) and np.isfinite(a).all() and np.isfinite(b).all()):
                continue
            dist = np.linalg.norm(a-b)
            if dist > .45:
                continue
            u = projected_offset(a,poses[1],ks[1],boxes[1][j])
            v = projected_offset(b,poses[0],ks[0],boxes[0][i])
            if u is not None and v is not None:
                cost[i,j] = .5 * (u+v) + dist
    for i,j in zip(*np.where(np.isfinite(cost))):
        alternatives = np.r_[np.delete(cost[i,:],j),np.delete(cost[:,j],i)]
        if len(alternatives) and cost[i,j] + .15 >= alternatives.min():
            continue
        out["matches"].append((int(i),int(j)))
    for (i,j),(k,l) in combinations(out["matches"],2):
        if (min(i,k),max(i,k)) in sep[0] and (min(j,l),max(j,l)) in sep[1]:
            out["distinct"].extend([(i,l),(k,j)])
    return out
