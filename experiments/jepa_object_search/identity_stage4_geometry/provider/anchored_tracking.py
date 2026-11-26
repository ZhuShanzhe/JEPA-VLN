"""Localize an RGB in an already observed map; only flow-verified actual points get depth."""
import cv2
import numpy as np


def estimate_from_anchor(native_rgb, view, live_rgb):
    H, W = native_rgb.shape[:2]
    h, w = live_rgb.shape[:2]
    record = {"reason": "incompatible_camera"}
    if H*w != W*h or native_rgb.shape[2:] != (3,) or live_rgb.shape[2:] != (3,):
        return None, record
    # Solve in native coordinates, return in the raw live pixel coordinate system.
    anchor = cv2.cvtColor(native_rgb, cv2.COLOR_RGB2GRAY)
    live = cv2.resize(cv2.cvtColor(live_rgb, cv2.COLOR_RGB2GRAY), (W, H), interpolation=cv2.INTER_AREA)
    yy, xx = np.indices(view["grid"].shape[:2])
    stride = view["stride"]
    source_xy = np.column_stack([xx.ravel()*stride, yy.ravel()*stride]).astype(np.float32)
    xyz = view["grid"].reshape(-1,3)
    anchor_pose, K = view["c2w"], view["K"]
    z = (xyz-anchor_pose[:3,3]) @ anchor_pose[:3,2]
    valid = (np.isfinite(xyz).all(axis=1) & (z>.05) & (z<30)
             & (source_xy[:,0]>8) & (source_xy[:,0]<W-9) & (source_xy[:,1]>8) & (source_xy[:,1]<H-9))
    points, source_xy = xyz[valid], source_xy[valid]
    if len(points)<40:
        return None, {"reason":"few_map_points"}
    opts = dict(winSize=(19,19), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    target, fw, ferr = cv2.calcOpticalFlowPyrLK(anchor, live, source_xy, None, **opts)
    back, bw, _ = cv2.calcOpticalFlowPyrLK(live, anchor, target, None, **opts)
    good = (fw.ravel().astype(bool) & bw.ravel().astype(bool) & (ferr.ravel()<15)
            & (np.linalg.norm(back-source_xy, axis=1)<.8)
            & (target[:,0]>4) & (target[:,0]<W-5) & (target[:,1]>4) & (target[:,1]<H-5))
    # Offset .5 matches the map-view K and the frozen identity head's pixel-center convention.
    source_xy, points, xy = source_xy[good], points[good], target[good]+.5
    record={"tracked":int(len(points))}
    def fail(reason):
        return None, dict(record,reason=reason)
    if len(points)<45:
        return fail("few_bidirectional_tracks")
    # Hold out deterministic source-grid cells before fitting. Neighbouring tracks can
    # be correlated; also require image-wide coverage of accepted correspondences.
    cell = (source_xy//stride).astype(int)
    hold = ((cell[:,0]+2*cell[:,1])%5)==0
    if hold.sum()<10 or (~hold).sum()<32:
        return fail("few_holdout_tracks")
    ok,r,t,ids = cv2.solvePnPRansac(np.ascontiguousarray(points[~hold]),np.ascontiguousarray(xy[~hold].astype(float)),K,None,
                                    iterationsCount=150,reprojectionError=2.5,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or ids is None or len(ids)<24:
        return fail("pnp_ransac_rejected")
    train = np.flatnonzero(~hold)[ids.ravel()]
    r,t = cv2.solvePnPRefineLM(points[train],xy[train].astype(float),K,None,r,t)
    proj = cv2.projectPoints(points,r,t,K,None)[0].reshape(-1,2)
    err = np.linalg.norm(proj-xy, axis=1)
    stable = np.isfinite(err) & (err<=2.5)
    fraction=float(stable[hold].mean())
    hull = cv2.convexHull(xy[train].astype(np.float32))
    area=float(cv2.contourArea(hull)/(W*H))
    median=float(np.median(err[hold]))
    record.update(train_inliers=int(len(train)),holdout_count=int(hold.sum()),holdout_inlier_fraction=round(fraction,3),
                  area_fraction=round(area,3),median_holdout_px=round(median,3) if np.isfinite(median) else None)
    if fraction<.70 or median>1.5 or area<.12:
        return fail("holdout_or_coverage_rejected")
    R=cv2.Rodrigues(r)[0]
    pose=np.eye(4)
    pose[:3,:3]=R.T
    pose[:3,3]=(-R.T@t).ravel()
    displacement=np.linalg.norm(pose[:3,3]-anchor_pose[:3,3])
    if not np.isfinite(displacement) or displacement>2.5:
        return fail("nonlocal_pose")
    live_z=(points-pose[:3,3])@pose[:3,2]
    keep=stable & (live_z>.05) & (live_z<30)
    # Multiple depths projecting into a pixel use the nearest. Never interpolate holes.
    pixels=np.floor(xy[keep]*[w/W,h/H]).astype(int)
    depth=np.full((h,w),np.inf,np.float32)
    np.minimum.at(depth,(pixels[:,1],pixels[:,0]),live_z[keep].astype(np.float32))
    depth[~np.isfinite(depth)]=np.nan
    scaled_K=K.copy()
    scaled_K[:2]*=np.array([w/W,h/H])[:,None]
    record.update(reason="localized",displacement_m=round(float(displacement),3),depth_pixels=int(np.isfinite(depth).sum()))
    return (depth,scaled_K,pose),record
