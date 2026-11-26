"""Carry mapped 3-D points through actual intermediate RGB frames, then fit the current pose."""
import cv2
import numpy as np


def chain_geometry(view, gray_frames, output_hw):
    if len(gray_frames)<2:
        return None,"no_interval"
    H,W=gray_frames[0].shape
    h,w=output_hw
    grid=view["grid"]
    Y,X=np.indices(grid.shape[:2])
    pixel=np.stack((X,Y),axis=-1).reshape(-1,2).astype(np.float32)*view["stride"]
    xyz=grid.reshape(-1,3)
    z=(xyz-view["c2w"][:3,3])@view["c2w"][:3,2]
    valid=np.isfinite(xyz).all(1)&(z>.05)&(z<30)&(pixel[:,0]>8)&(pixel[:,0]<W-9)&(pixel[:,1]>8)&(pixel[:,1]<H-9)
    coords=pixel[valid]
    cell=(coords//view["stride"]).astype(int)
    points=xyz[valid]
    hold=(cell[:,0]+2*cell[:,1])%5==0
    opts=dict(winSize=(19,19),maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    for first,second in zip(gray_frames,gray_frames[1:]):
        if len(coords)<45:
            return None,"few_tracked_map_points"
        nxt,ok,err=cv2.calcOpticalFlowPyrLK(first,second,coords,None,**opts)
        back,ok2,_=cv2.calcOpticalFlowPyrLK(second,first,nxt,None,**opts)
        keep=(ok[:,0].astype(bool)&ok2[:,0].astype(bool)&(err[:,0]<15)&(np.linalg.norm(back-coords,axis=1)<.8)
              &(nxt[:,0]>4)&(nxt[:,0]<W-5)&(nxt[:,1]>4)&(nxt[:,1]<H-5))
        coords,hold,points=nxt[keep],hold[keep],points[keep]
    if hold.sum()<10 or (~hold).sum()<32:
        return None,"too_few_final_tracks"
    uv=coords.astype(float)+.5
    K=view["K"]
    unskew=uv.copy()
    unskew[:,0]-=K[0,1]*(uv[:,1]-K[1,2])/K[1,1]
    pinhole=K.copy()
    pinhole[0,1]=0
    ok,r,t,ids=cv2.solvePnPRansac(points[~hold],unskew[~hold],pinhole,None,iterationsCount=150,
                                 reprojectionError=2.5,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or ids is None or len(ids)<24:
        return None,"pnp_rejected"
    fit=np.flatnonzero(~hold)[ids.ravel()]
    r,t=cv2.solvePnPRefineLM(points[fit],unskew[fit],pinhole,None,r,t)
    proj=cv2.projectPoints(points,r,t,pinhole,None)[0].reshape(-1,2)
    proj[:,0]+=K[0,1]*(proj[:,1]-K[1,2])/K[1,1]
    residual=np.linalg.norm(proj-uv,axis=1)
    hull=cv2.contourArea(cv2.convexHull(uv[fit].astype(np.float32)))/(W*H)
    if (residual[hold]<=2.5).mean()<.70 or np.median(residual[hold])>1.5 or hull<.12:
        return None,"heldout_or_coverage_rejected"
    pose=np.eye(4)
    pose[:3,:3]=cv2.Rodrigues(r)[0].T
    pose[:3,3]=(-pose[:3,:3]@t).ravel()
    if np.linalg.norm(pose[:3,3]-view["c2w"][:3,3])>2.5:
        return None,"nonlocal_pose"
    depth_z=(points-pose[:3,3])@pose[:3,2]
    good=(residual<=2.5)&(depth_z>.05)&(depth_z<30)
    pix=np.floor(uv[good]*[w/W,h/H]).astype(int)
    depth=np.full((h,w),np.inf,np.float32)
    np.minimum.at(depth,(pix[:,1],pix[:,0]),depth_z[good].astype(np.float32))
    depth[~np.isfinite(depth)]=np.nan
    outK=K.copy()
    outK[:2]*=np.array([w/W,h/H])[:,None]
    return (depth,outK,pose),"localized"
