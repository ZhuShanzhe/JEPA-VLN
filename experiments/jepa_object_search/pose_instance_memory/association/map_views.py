"""Estimate the pinhole intrinsics of a mapped frame from its observed point grid."""
import numpy as np


def metric_view(snapshot, scale, stride):
    pts=np.asarray(snapshot["points"],float)
    rows=np.asarray(snapshot["rows"])
    pose=np.array(snapshot["pose"],float,copy=True)
    if pose.shape!=(4,4) or pts.ndim!=2 or pts.shape[1]!=3 or len(rows)!=len(pts) or len(rows)<50:
        return None
    h=len(np.unique(rows))
    if h<3 or len(rows)%h:
        return None
    w=len(rows)//h
    if not np.array_equal(rows,np.repeat(np.arange(h)*stride,w)):
        return None
    if not np.isfinite(pose).all() or not np.isfinite(scale) or scale<=0:
        return None
    u,s,v=np.linalg.svd(pose[:3,:3])
    if s.min()<=0 or s.max()/s.min()>1.02 or np.linalg.det(u@v)<0:
        return None
    pose[:3,:3]=u@v
    metric=pts*scale
    pose[:3,3]*=scale
    local=(metric-pose[:3,3]) @ pose[:3,:3]
    ok=np.isfinite(local).all(1) & (local[:,2]>.05)
    if ok.sum()<40:
        return None
    rays=local[ok,:2]/local[ok,2:3]
    xy=np.stack((np.tile(np.arange(w)*stride,h),rows),axis=1)[ok]+.5
    K=np.eye(3)
    for axis in range(2):
        if np.ptp(rays[:,axis])<.15:
            return None
        design=np.column_stack((rays[:,axis],np.ones(len(rays))))
        fit=np.linalg.lstsq(design,xy[:,axis],rcond=None)[0]
        if fit[0]<=0 or np.quantile(np.abs(design@fit-xy[:,axis]),.9)>2.5:
            return None
        K[axis,axis], K[axis,2]=fit
    return dict(c2w=pose,K=K,grid=metric.reshape(h,w,3),stride=stride,frame_id=snapshot["frame_id"])


def point_in_box(view,box):
    grid,stride=view["grid"],view["stride"]
    a,b=np.asarray(box[:2],float),np.asarray(box[2:],float)
    margin=.15*(b-a)
    x0,y0=np.maximum(np.ceil((a+margin)/stride).astype(int),0)
    x1,y1=np.minimum(np.ceil((b-margin)/stride).astype(int),grid.shape[1::-1])
    if x0>=x1 or y0>=y1:
        return None
    pts=grid[y0:y1,x0:x1].reshape(-1,3)
    pts=pts[np.isfinite(pts).all(1)]
    if len(pts)<3:
        return None
    depth=(pts-view["c2w"][:3,3])@view["c2w"][:3,2]
    ok=depth>.05
    if ok.sum()<3:
        return None
    pts,depth=pts[ok],depth[ok]
    med=np.median(depth)
    support=pts[np.abs(depth-med)<=max(.04,.025*med)]
    if not len(support):
        return None
    return support[np.linalg.norm(support-np.median(support,axis=0),axis=1).argmin()].tolist()


def rgb_aligned(snapshot, rgb, stride):
    """Check normalized detector clicks against native map color geometry."""
    colors=snapshot.get("colors")
    if colors is None:
        return False
    h=len(np.unique(snapshot["rows"]))
    if not h or len(colors)%h or h<4:
        return False
    w=len(colors)//h
    # Grid dimensions may end on a partial stride. Use the documented image
    # coordinate conversion only when independent grid colors agree with it.
    H,W=rgb.shape[:2]
    yy=(np.arange(h)*H/h).round().astype(int).clip(0,H-1)
    xx=(np.arange(w)*W/w).round().astype(int).clip(0,W-1)
    sample=rgb[np.ix_(yy,xx)].astype(float)
    observed=np.asarray(colors,float).reshape(h,w,3)
    if observed.std(axis=(0,1)).max()<10:
        return False
    return bool(np.abs(sample-observed).mean()<12.)
