"""Diagnose the projection of native projective-SLAM world points onto their own pixels."""
import cv2
import numpy as np


def skew_view(snap,scale,stride=8):
    """Recover the native upper-triangular K at its exported, fixed camera pose."""
    pts=np.asarray(snap["points"],float)*scale
    ys=np.asarray(snap["rows"])
    h=len(np.unique(ys))
    if h<4 or len(ys)%h:return None
    w=len(ys)//h
    if not np.array_equal(ys,np.repeat(np.arange(h)*stride,w)):
        return None
    raw=np.array(snap["pose"],float)
    if raw.shape!=(4,4) or not np.isfinite(raw).all():return None
    u,s,vh=np.linalg.svd(raw[:3,:3])
    if s.min()<=0 or s.max()/s.min()>1.02 or np.linalg.det(u@vh)<0:return None
    raw[:3,:3]=u@vh
    raw[:3,3]*=scale
    cam=(pts-raw[:3,3])@raw[:3,:3]
    valid=np.isfinite(cam).all(1)&(cam[:,2]>.05)
    if valid.sum()<80:return None
    ij=np.indices((h,w)).reshape(2,-1).T[:,::-1][valid]
    rays=cam[valid,:2]/cam[valid,2:3]
    xy=ij*stride+.5
    hold=(ij[:,0]+ij[:,1]*2)%5==0
    if hold.sum()<15 or (~hold).sum()<48:return None
    design=np.column_stack((rays,np.ones(len(rays))))
    K=np.eye(3)
    K[0]=np.linalg.lstsq(design[~hold],xy[~hold,0],rcond=None)[0]
    K[1,1:]=np.linalg.lstsq(design[~hold,1:],xy[~hold,1],rcond=None)[0]
    err=np.linalg.norm(design[hold]@K[:2].T-xy[hold],axis=1)
    if min(K[0,0],K[1,1])<=0 or abs(K[0,1])>.15*K[0,0] or np.quantile(err,.9)>2.5:
        return None
    return dict(c2w=raw,K=K,grid=pts.reshape(h,w,3),stride=stride,frame_id=int(snap["frame_id"]))


def fit_camera(xyz,xy):
    c=xyz.mean(0)
    s=np.sqrt(3)/max(np.linalg.norm(xyz-c,axis=1).mean(),1e-8)
    n=np.eye(4); n[:3,:3]*=s; n[:3,3]=-s*c
    m=xy.mean(0)
    t=np.sqrt(2)/max(np.linalg.norm(xy-m,axis=1).mean(),1e-8)
    l=np.diag([t,t,1.]);l[:2,2]=-t*m
    X=np.column_stack(((xyz-c)*s,np.ones(len(xyz))))
    u,v=((xy-m)*t).T
    z=np.zeros_like(X)
    A=np.concatenate((np.column_stack((X,z,-u[:,None]*X)),np.column_stack((z,X,-v[:,None]*X))))
    _,_,vt=np.linalg.svd(A,full_matrices=False)
    P=np.linalg.inv(l)@vt[-1].reshape(3,4)@n
    if np.linalg.det(P[:,:3])<0:
        P=-P
    K,R,T,*_=cv2.decomposeProjectionMatrix(P)
    pose=np.eye(4);pose[:3,:3]=R.T;pose[:3,3]=(T[:3]/T[3]).ravel()
    return P,K/K[2,2],pose


def diagnose(snap,native,scale):
    h,w=native.shape[:2]; pts=np.asarray(snap["points"],float)*scale
    ys=snap["rows"];H=len(np.unique(ys));W=len(ys)//H
    ij=np.indices((H,W)).reshape(2,-1).T[:,::-1];uv=ij*8+.5
    ok=np.isfinite(pts).all(1)
    p,xy,ij=pts[ok],uv[ok],ij[ok]
    stats={"finite":int(len(p))}
    if len(p)<80:return None,stats
    hold=(ij[:,0]+2*ij[:,1])%5==0
    if hold.sum()<15 or (~hold).sum()<48:return None,stats
    def pixerr(P):
        a=np.column_stack((p,np.ones(len(p))))@P.T
        return np.linalg.norm(a[:,:2]/a[:,2:]-xy,axis=1)
    P,K,pose=fit_camera(p[~hold],xy[~hold])
    err=pixerr(P)
    # Trim only gross errors inside the fitting fold; leave hold-out points untouched.
    fit=(~hold)&(err < max(1.5,2.5*np.median(err[~hold])))
    if fit.sum()>=48:
        P,K,pose=fit_camera(p[fit],xy[fit]);err=pixerr(P)
    native_pose=np.array(snap["pose"],float);native_pose[:3,3]*=scale
    R0=native_pose[:3,:3]; z=(p-native_pose[:3,3])@R0[:,2]
    local=(p-native_pose[:3,3])@R0
    mask=(z>.05)
    naive=[]
    for axis in (0,1):
        ray=local[mask,axis]/local[mask,2]
        fit1=np.c_[ray,np.ones(mask.sum())]
        co=np.linalg.lstsq(fit1,xy[mask,axis],rcond=None)[0]
        naive.append([*co.tolist(),float(np.quantile(abs(fit1@co-xy[mask,axis]),.9))])
    stats.update(K=K.round(2).tolist(),positive_native_fraction=round(float(mask.mean()),3),native_linear_fit=np.round(naive,3).tolist(),
                 holdout_p50_px=round(float(np.median(err[hold])),3),holdout_p90_px=round(float(np.quantile(err[hold],.9)),3),
                 relative_origin_m=round(float(np.linalg.norm(pose[:3,3]-native_pose[:3,3])),3),
                 relative_rotation_deg=round(float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3].T@R0)-1)/2,-1,1)))),3))
    return (K,pose),stats
