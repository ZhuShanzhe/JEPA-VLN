"""Associate a registered object with a live detection at an independently estimated arrival."""
import cv2
import numpy as np
from visual_witness import compare_regions, corroborated_surroundings, describe, strong_object

SIZE = 360
SIDE = 96
LO = (SIZE-SIDE)//2
HI = LO+SIDE


def _region(rgb, box):
    rgb = np.asarray(rgb)
    box = np.asarray(box, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.dtype != np.uint8 or box.shape != (4,):
        return None
    h,w = rgb.shape[:2]
    if (not np.isfinite(box).all() or min(box[:2])<1 or box[2]>w-1 or box[3]>h-1
            or min(box[2:]-box[:2]) < 10):
        return None
    sx,sy = SIDE/(box[2:]-box[:2])
    affine = np.float32([[sx,0,LO-sx*box[0]],[0,sy,LO-sy*box[1]]])
    tile=cv2.warpAffine(rgb[:,:,:3],affine,(SIZE,SIZE),borderMode=cv2.BORDER_REPLICATE)
    gray=cv2.cvtColor(tile,cv2.COLOR_RGB2GRAY)
    valid=cv2.warpAffine(np.full((h,w),255,np.uint8),affine,(SIZE,SIZE),flags=cv2.INTER_NEAREST)
    valid=cv2.erode(valid,np.ones((11,11),np.uint8))
    key,des=cv2.SIFT_create(nfeatures=900,contrastThreshold=.012).detectAndCompute(gray,valid)
    if des is None:
        return None
    points=np.float32([k.pt for k in key])
    return gray,points,des


def _matches(a,b,zone):
    aid=np.flatnonzero(zone(a[1]))
    bid=np.flatnonzero(zone(b[1]))
    if min(len(aid),len(bid))<2:
        return np.empty((0,2,2),np.float32)
    bf=cv2.BFMatcher()
    def ratios(x,y):
        return {m.queryIdx:m.trainIdx for m,n in bf.knnMatch(x,y,k=2) if m.distance < .8*n.distance}
    left,right=a[2][aid],b[2][bid]
    lr,rl=ratios(left,right),ratios(right,left)
    pairs=[(a[1][aid[i]],b[1][bid[j]]) for i,j in lr.items() if rl.get(j)==i]
    pairs=np.float32(pairs).reshape(-1,2,2)
    if len(pairs):
        _,idx=np.unique(np.rint(pairs.reshape(-1,4)/3),axis=0,return_index=True)
        pairs=pairs[idx]
    return pairs


def prepare_reference(rgb, box):
    return _region(rgb, box), describe(rgb, box)


def _agrees(ref, query):
    if ref is None or query is None:
        return False
    ring=_matches(ref,query,lambda p:np.any((p<LO-5)|(p>HI+5),axis=1))
    core=_matches(ref,query,lambda p:np.all((p>LO+2)&(p<HI-2),axis=1))
    if len(ring)<5:
        return False
    # Find the projected object location from the surroundings alone. Mask
    # matches inside the object so identical artwork cannot certify its own location.
    a,b=ring[:,0],ring[:,1]
    H,mask=cv2.findHomography(a,b,cv2.RANSAC,4.0)
    if H is None or mask is None or not np.isfinite(H).all():
        return False
    keep=mask.ravel().astype(bool)
    if keep.sum()<5 or keep.sum()<.45*len(keep):
        return False
    center=np.float32([SIZE/2,SIZE/2])
    # Near-collinear keypoints on a remote door frame cannot predict the object.
    for points in (a[keep],b[keep]):
        hull=cv2.convexHull(points)
        if (cv2.contourArea(hull)<SIDE**2*.30 or
                cv2.pointPolygonTest(hull,(float(center[0]),float(center[1])),True)<-SIDE*.72):
            return False
    def project(points):
        return cv2.perspectiveTransform(np.float32(points).reshape(-1,1,2),H).reshape(-1,2)
    corners=project([[LO,LO],[HI,LO],[HI,HI],[LO,HI]])
    if (not cv2.isContourConvex(corners) or cv2.contourArea(corners,oriented=True)<=0 or
            not .42*SIDE**2<=cv2.contourArea(corners)<=2.3*SIDE**2 or
            np.linalg.norm(project(center[None])[0]-center)>SIDE*.18):
        return False
    # Independently verify the object's content under the background-predicted
    # warp. A similar room alone cannot satisfy the gate.
    predicted=cv2.warpPerspective(ref[0],H,(SIZE,SIZE))
    sl=np.s_[LO+7:HI-7,LO+7:HI-7]
    p=cv2.GaussianBlur(predicted,(7,7),1.8)[sl].astype(float)
    q=cv2.GaussianBlur(query[0],(7,7),1.8)[sl].astype(float)
    if min(p.std(),q.std())<8:
        return False
    # Compare local contrast too. Shared silhouettes and a clock's hands can
    # give two distinct prints a high correlation at low frequencies.
    detail_a=p-cv2.GaussianBlur(p,(0,0),4)
    detail_b=q-cv2.GaussianBlur(q,(0,0),4)
    band_corr=float(np.sum(detail_a*detail_b)/max(1.,np.linalg.norm(detail_a)*np.linalg.norm(detail_b)))
    nr,nc=detail_a.shape
    tiles=[]
    for y in np.array_split(np.arange(nr),3):
        for x in np.array_split(np.arange(nc),3):
            region=np.ix_(y,x)
            aa,bb=detail_a[region],detail_b[region]
            if min(aa.std(),bb.std())>2.5:
                tiles.append(float(np.sum(aa*bb)/max(1.,np.linalg.norm(aa)*np.linalg.norm(bb))))
    if len(tiles)<3 or np.median(tiles)<.68 or band_corr<.64:
        return False
    p-=p.mean()
    q-=q.mean()
    correlation=np.sum(p*q)/max(1.,np.linalg.norm(p)*np.linalg.norm(q))
    core_inliers=0
    if len(core):
        error=np.linalg.norm(project(core[:,0])-core[:,1],axis=1)
        core_inliers=int(np.count_nonzero(error<7))
    return bool((correlation>=.80 and core_inliers>=2) or
                (correlation>=.89 and keep.sum()>=8))


def choose_arrival_box(reference_rgb, reference_xyxy, current_rgb, current_boxes):
    """Return a unique proposal index or None. Boxes are native RGB pixel coordinates."""
    return match_prepared(prepare_reference(reference_rgb, reference_xyxy),current_rgb,current_boxes)


def match_prepared(ref,current_rgb,current_boxes):
    surroundings,object_region=ref
    matches=[]
    for i,box in enumerate(current_boxes):
        # A protruding object can disagree with the background-plane warp.
        # Also allow the original object-led matcher if *both* the interior
        # and neighboring landmarks corroborate the match on this same image.
        if _agrees(surroundings,_region(current_rgb,box)):
            matches.append(i)
            continue
        stats=compare_regions(object_region,describe(current_rgb,box))
        if strong_object(stats) and corroborated_surroundings(stats):
            matches.append(i)
    return matches[0] if len(matches)==1 else None
