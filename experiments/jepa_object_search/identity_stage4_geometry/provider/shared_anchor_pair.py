"""Public, filesystem-independent two-anchor geometry bridge.

Inputs are RGB images, monocular VGGT output, already-published map views and a
frozen verifier. Navigation coordinates, episode names and simulator state are
never accessed here. All pose matrices are OpenCV camera-to-world. Timestamps
and causality of the supplied eight-image window are the caller's responsibility.
"""
import cv2
import numpy as np


def camera_frame(world_depth, intrinsics, camera_to_world, stride=8):
    y,x=np.mgrid[0:world_depth.shape[0]:stride,0:world_depth.shape[1]:stride]
    rays=np.stack([x+.5,y+.5,np.ones_like(x)],axis=-1)@np.linalg.inv(intrinsics).T
    return (rays*world_depth[::stride,::stride,None])@camera_to_world[:3,:3].T+camera_to_world[:3,3]


def windows_with_anchors(query_step,anchor_steps):
    """Eight causal images including the specified two known mapping anchors."""
    if len(set(anchor_steps))!=2 or min(anchor_steps)<0 or max(anchor_steps)>=query_step or query_step<8:
        raise ValueError("expected two distinct previously observed anchors and at least eight past images")
    prior=list(range(query_step-8,query_step))
    if all(k in prior for k in anchor_steps):
        return prior
    start=min(query_step-8,*anchor_steps)
    selected=set([*anchor_steps,query_step-1])
    while len(selected)<8:
        candidates=[k for k in range(start,query_step) if k not in selected]
        selected.add(max(candidates,key=lambda k:(min(abs(k-y) for y in selected),k)))
    return sorted(selected)


def verified_image_edges(images,depth,K,poses):
    """Check that predicted 3-D geometry explains independent RGB feature matches."""
    sift=cv2.SIFT_create(nfeatures=1400)
    features=[sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),None) for im in images]
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    def good(a,b):
        return {m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(a,b,k=2) if m.distance<.78*n.distance}
    edges=[]
    for i,(a,fa) in enumerate(features):
        for j in range(i+1,len(images)):
            b,fb=features[j]
            if fa is None or fb is None or min(len(fa),len(fb))<12:continue
            f,g=good(fa,fb),good(fb,fa)
            ij=[(k,v) for k,v in f.items() if g.get(v)==k]
            if len(ij)<12:continue
            uv=np.array([a[k].pt for k,v in ij]);vt=np.array([b[v].pt for k,v in ij])
            ix=np.rint(uv[:,0]).astype(int).clip(0,depth.shape[2]-1)
            iy=np.rint(uv[:,1]).astype(int).clip(0,depth.shape[1]-1)
            rays=np.column_stack((uv+.5,np.ones(len(uv))))@np.linalg.inv(K[i]).T
            world=(rays*depth[i,iy,ix,None])@poses[i,:3,:3].T+poses[i,:3,3]
            local=(world-poses[j,:3,3])@poses[j,:3,:3]
            pix=local@K[j].T
            err=np.linalg.norm(pix[:,:2]/pix[:,2:]-(vt+.5),axis=1)
            keep=(local[:,2]>.05)&(err<=3.)
            if keep.sum()<12:continue
            area=cv2.contourArea(cv2.convexHull(uv[keep].astype(np.float32)))/np.prod(images[i].shape[:2])
            if area>=.04:edges.append([i,j,int(keep.sum()),round(float(area),3)])
    return edges


def split_camera_height_scales(depth,K,poses,navigator,height_m):
    """Use the caller's original mapping API with two disjoint past-frame subsets."""
    if len(poses)!=8 or not np.isfinite(height_m) or height_m<=0:
        raise ValueError("expected eight past frames and a calibrated camera height")
    frames=[dict(frame_id=i,points=camera_frame(d,k,t).reshape(-1,3),pose=t) for i,(d,k,t) in enumerate(zip(depth,K,poses))]
    align=navigator.gravity_alignment(poses,cam_up=navigator.mount_compensated_cam_up())
    scales=[]
    for subset in (frames[::2],frames[1::2]):
        grid=navigator.OccupancyGrid.from_frame_points(subset,align,cam_height_m=height_m,unit_per_m=None)
        scales.append(float(1/grid.unit_per_m) if grid is not None and grid.unit_per_m>0 else None)
    return scales


def verify_pair(images,depth,K,poses,source_view,source_box,live_boxes,anchors,floor_scales,verifier,edges=None):
    """Return a diagnostic record and the frozen identity decision when scorable.

    images consists of the source, eight chronological past frames and the current
    frame in the local reconstruction's common pixel grid. The source RGB must
    be the original map grid. Each tuple in anchors is (frame_id, image_index,
    native_rgb, metric_view). A metric_view contains K, c2w and an 8-stride
    world XYZ grid in metres. Boxes are xyxy in this pixel grid. A verifier
    implements encode/score/decide. Missing evidence returns 'withheld'.
    """
    if len(images)!=10 or depth.shape[0]!=10 or len(K)!=10 or len(poses)!=10:
        raise ValueError("provide one registered source, eight past images, and the query")
    H,W=depth.shape[1:];last=len(poses)-1
    output=dict(anchors=[],result="withheld")
    def finish(reason):
        return dict(output,reason=reason)
    if source_view is None:return finish("invalid_source_or_scale")
    edges=verified_image_edges(images,depth,K,poses) if edges is None else edges
    graph=[set() for _ in poses]
    for i,j,*_ in edges:
        graph[i].add(j);graph[j].add(i)
    connected={last};q=[last]
    while q:
        n=q.pop();extra=graph[n]-connected;connected.update(extra);q.extend(extra)
    accepted=[]
    for fid,i,native,view in sorted(anchors,key=lambda r:r[0]):
        if i not in range(1,last):raise ValueError("shared anchor must be in the causal past")
        mae=float(np.abs(images[i].astype(float)-native).mean())
        rec=dict(frame=fid,mae=round(mae,3),connected=i in connected)
        output["anchors"].append(rec)
        if mae>=10 or view is None or i not in connected:continue
        local_grid=camera_frame(depth[i],K[i],poses[i])
        xyz=view["grid"]
        z=(xyz-view["c2w"][:3,3])@view["c2w"][:3,2]
        ok=np.isfinite(z)&(z>.05)&(z<30)
        ii,jj=np.indices(ok.shape)
        fit=ok&((ii+jj*2)%5!=0);hold=ok&~fit
        if min(fit.sum(),hold.sum())<40:continue
        s=float(np.median(z[fit]/depth[i,::8,::8][fit]))
        R=view["c2w"][:3,:3]@poses[i,:3,:3].T
        t=view["c2w"][:3,3]-s*R@poses[i,:3,3]
        rec["visual_unit_to_native_m"]=round(s,4)
        accepted.append((fid,i,s,R,t,local_grid,view,hold,z))
    if len(accepted)<2:return finish("fewer_than_two_shared_timestamped_anchors")
    scale,R,t=accepted[0][2:5]
    errors=[]
    for fid,i,s,_,_,grid,view,hold,z in accepted:
        projection=scale*grid[hold]@R.T+t
        rel=np.linalg.norm(projection-view["grid"][hold],axis=1)/z[hold]
        centre_error=np.linalg.norm(scale*R@poses[i,:3,3]+t-view["c2w"][:3,3])
        rot=(R@poses[i,:3,:3]).T@view["c2w"][:3,:3]
        angle=np.degrees(np.arccos(np.clip((np.trace(rot)-1)/2,-1,1)))
        errors.append(dict(frame=fid,holdout=int(hold.sum()),median_relative_3d_error=round(float(np.median(rel)),4),
                           p90_relative_3d_error=round(float(np.quantile(rel,.9)),4),centre_error_m=round(float(centre_error),3),
                           orientation_error_deg=round(float(angle),3),scale_ratio=round(s/scale,3)))
    output["anchor_audit"]=errors
    if not all(e["p90_relative_3d_error"]<=.15 and e["centre_error_m"]<=.2 and e["orientation_error_deg"]<=5.
               and .85<=e["scale_ratio"]<=1.15 for e in errors):
        return finish("cross_anchor_metric_disagreement")
    if not all(floor_scales) or max(floor_scales)/min(floor_scales)>1.15:
        return finish("scale_disagreement_with_floor")
    ratio=round(float(scale/np.mean(floor_scales)),3)
    output["independent_ruler_ratio"]=ratio
    if not .8<=ratio<=1.25:return finish("scale_disagreement_with_floor")
    qworld=camera_frame(depth[-1],K[-1],poses[-1],stride=1)
    hits=np.zeros((H,W),np.uint8)
    for i in sorted(connected-{0,last}):
        cam=(qworld-poses[i,:3,3])@poses[i,:3,:3]
        pix=cam@K[i].T
        uv=np.nan_to_num(pix[:,:,:2]/pix[:,:,2:]).clip(-2,max(H,W)+1)
        ix=uv[:,:,0].astype(int).clip(0,W-1);iy=uv[:,:,1].astype(int).clip(0,H-1)
        valid=(cam[:,:,2]>.05)&(uv[:,:,0]>=0)&(uv[:,:,0]<W)&(uv[:,:,1]>=0)&(uv[:,:,1]<H)
        valid&=(abs(depth[i,iy,ix]-cam[:,:,2])<.08*cam[:,:,2])
        valid&=(np.abs(images[i][iy,ix].astype(float)-images[-1]).mean(2)<15)
        hits+=valid.astype(np.uint8)
    dl=np.where(hits>=2,depth[-1]*scale,np.nan)
    counts=[int(np.isfinite(dl[int(y):int(np.ceil(v)),int(x):int(np.ceil(u))]).sum()) for x,y,u,v in live_boxes]
    output["live_corroborated_pixels_per_box"]=counts
    if not counts or min(counts)<9:return finish("missing_corroborated_depth_in_box")
    z=(source_view["grid"]-source_view["c2w"][:3,3])@source_view["c2w"][:3,2]
    sd=np.full(images[0].shape[:2],np.nan,np.float32);sd[::8,::8]=z
    pose=np.eye(4);pose[:3,:3]=R@poses[-1,:3,:3];pose[:3,3]=scale*R@poses[-1,:3,3]+t
    original=verifier.encode(images[0],[source_box],sd,source_view["K"],source_view["c2w"])
    current=verifier.encode(images[-1],live_boxes,dl,K[-1],pose)
    scores=verifier.score(original,current);decision,=verifier.decide(scores)
    output.update(result=decision["state"],selected_box=decision["index"],
                  scores=[float(x) if np.isfinite(x) else None for x in scores.ravel()])
    return finish("frozen_verifier_scored")
