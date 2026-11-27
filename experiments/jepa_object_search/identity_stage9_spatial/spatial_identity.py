"""Conservative instance association from two object-centred, metric map observations.

All anchors compared together must be reconstructed in the same map revision.
The camera center is used only to check independence and orient a surface.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Surface:
    point: tuple
    normal: tuple


@dataclass(frozen=True)
class Anchor:
    frame_id: int
    center: tuple
    camera: tuple
    sigma: float
    surface: Surface | None
    hue: float | None = None
    clipped: bool = False
    dimensions: tuple | None = None


@dataclass(frozen=True)
class Decision:
    status: str
    owner: object
    reason: str


def _dominant_hue(rgb, box):
    """Use a distinctive saturated foreground only if uncommon in the surrounding ring."""
    h,w=rgb.shape[:2]
    x,y,u,v=box
    a,b=(u-x)*.20,(v-y)*.20
    def hues(crop):
        pixels=crop.reshape(-1,3).astype(float)
        if not len(pixels):
            return np.array([])
        red,green,blue=pixels.T
        chroma=pixels.max(1)-pixels.min(1)
        angles=np.degrees(np.arctan2(3**.5*(green-blue),2*red-green-blue)) % 360
        return np.where((chroma>=42) & (chroma>=.32*pixels.max(1)),angles,np.nan)
    inner=rgb[int(y+b):math.ceil(v-b),int(x+a):math.ceil(u-a):1]
    values=hues(inner[::2,::2])
    valid=values[np.isfinite(values)]
    if len(valid)<20 or len(valid)<.55*len(values):
        return None
    vector=np.exp(1j*np.radians(valid)).mean()
    if abs(vector)<.94:
        return None
    hue=float(np.degrees(np.angle(vector))%360)
    left,top=max(0,int(x-a)),max(0,int(y-b))
    right,bottom=min(w,math.ceil(u+a)),min(h,math.ceil(v+b))
    rows,cols=np.mgrid[top:bottom:2,left:right:2]
    outside=(rows<y)|(rows>v)|(cols<x)|(cols>u)
    ring=hues(rgb[rows[outside],cols[outside]])
    if len(ring)<20 or np.mean(np.abs((ring-hue+180)%360-180)<28)>.20:
        return None
    return hue


def _surface(points,center,camera):
    """Fit a nearby observed surface around the box, not the camera-facing box plane."""
    if len(points)<12:
        return None
    # Keep bounded, deterministic work at all input resolutions.
    if len(points)>320:
        points=points[np.linspace(0,len(points)-1,320,dtype=int)]
    rng=np.random.default_rng(431)
    sample=rng.integers(len(points),size=(96,3))
    p,q,r=points[sample].transpose((1,0,2))
    normals=np.cross(q-p,r-p)
    norms=np.linalg.norm(normals,axis=1)
    valid=norms>1e-5
    if not valid.any():
        return None
    normals=normals[valid]/norms[valid,None]
    offsets=np.einsum('ij,ij->i',p[valid],normals)
    feasible=np.abs(normals@center-offsets)<.35
    if not feasible.any():
        return None
    normals,offsets=normals[feasible],offsets[feasible]
    masks=np.abs(points@normals.T-offsets)<.045
    i=int(np.argmax(masks.sum(axis=0)))
    support=points[masks[:,i]]
    if len(support)<max(12,.48*len(points)):
        return None
    origin=np.median(support,axis=0)
    _,s,vt=np.linalg.svd(support-origin,full_matrices=False)
    normal=vt[-1]
    if s[1]/math.sqrt(len(support))<.035 or s[2]>s[1]*.40:
        return None
    if abs(np.dot(center-origin,normal))>.35:
        return None
    if np.dot(camera-origin,normal)<0:
        normal=-normal
    return Surface(tuple(origin),tuple(normal))


def _clear_silhouette(rgb,box):
    """Require a simple background and visible contrasting material near all four box edges."""
    x,y,u,v=box
    h,w=rgb.shape[:2]
    pad=max(3,int(.14*min(u-x,v-y)))
    a,b,c,d=int(x),int(y),math.ceil(u),math.ceil(v)
    if min(a-pad,b-pad,w-c-pad,h-d-pad)<0 or min(c-a,d-b)<18:
        return False
    # Bound work on close-ups while retaining all four edges.
    inc=max(1,math.ceil(max(u-x,v-y)/150))
    rows,cols=np.mgrid[b-pad:d+pad:inc,a-pad:c+pad:inc]
    outside=(rows<b)|(rows>=d)|(cols<a)|(cols>=c)
    pixels=rgb[rows,cols].astype(float)
    ring=pixels[outside]
    if len(ring)<12:
        return False
    bg=np.median(ring,axis=0)
    ring_contrast=np.max(np.abs(ring-bg),axis=1)
    if np.percentile(ring_contrast,82)>23:
        return False
    fore=np.max(np.abs(pixels-bg),axis=-1)>max(34.,2*np.percentile(ring_contrast,75))
    inset=~outside
    for band in (cols<x+.14*(u-x),cols>u-.14*(u-x),
                 rows<y+.14*(v-y),rows>v-.14*(v-y)):
        strip=fore[inset & band]
        if strip.size<5 or strip.mean()<.18:
            return False
    return True


def _dimensions(plane,rgb,box,lo,hi,camera,center,rotation,K,stride,clipped,sigma):
    """Estimate near-frontal extents in metres in a plane parallel to the observed support."""
    if plane is None or clipped or sigma>.11 or min(hi-lo)<5*stride:
        return None
    n=np.asarray(plane.normal)
    ray=center-camera
    if (abs((center-plane.point)@n)>.10 or np.linalg.norm(ray)<.15 or
            -(ray@n)/np.linalg.norm(ray)<.90 or not _clear_silhouette(rgb,box)):
        return None
    xy=[[lo[0],lo[1],1],[hi[0],lo[1],1],[hi[0],hi[1],1],[lo[0],hi[1],1]]
    rays=np.linalg.solve(K,np.asarray(xy).T).T@rotation.T
    den=rays@n
    if np.any(den>=-.1):
        return None
    corners=camera+rays*((ray@n)/den)[:,None]
    edges=np.linalg.norm(np.roll(corners,-1,axis=0)-corners,axis=1)
    if np.max(np.abs(edges[:2]-edges[2:])/np.maximum(edges[:2],1e-6))>.20:
        return None
    sides=np.sort(.5*(edges[:2]+edges[2:]))
    return tuple(sides) if sides[0]>.08 else None


def make_anchor(view,rgb,box,*,frame_id=None):
    """Recover native-box center from the registered map depth; scale is already metric."""
    if view is None:
        return None
    rgb=np.asarray(rgb)
    box=np.asarray(box,dtype=float)
    grid=np.asarray(view['grid'],dtype=float)
    pose=np.asarray(view['c2w'],dtype=float)
    K=np.asarray(view['K'],dtype=float)
    if (rgb.ndim!=3 or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8 or grid.ndim!=3
            or grid.shape[-1]!=3 or pose.shape!=(4,4) or K.shape!=(3,3)
            or not np.isfinite(K).all() or not np.isfinite(pose).all()
            or min(K[0,0],K[1,1])<1 or box.shape!=(4,) or not np.isfinite(box).all()):
        return None
    h,w=rgb.shape[:2]
    x,y,u,v=box
    if min(x,y)<0 or u>w or v>h or min(u-x,v-y)<6:
        return None
    stride=float(view['stride'])
    gh,gw=grid.shape[:2]
    if not np.isfinite(stride) or stride<=0 or min(gh,gw,h,w)==0:
        return None
    scaled=box*np.tile([gw*stride/w,gh*stride/h],2)
    lo,hi=scaled[:2],scaled[2:]
    extent=hi-lo
    # Work in map-grid pixels, clipping the sampled ring at the real image edge.
    # Bound a narrow detection's ring to contain several actual depth samples.
    pad=np.maximum(.42*extent,3*stride)
    ij0=np.maximum(np.floor((lo-pad)/stride).astype(int),0)
    ij1=np.minimum(np.ceil((hi+pad)/stride).astype(int)+1,[gw,gh])
    yy,xx=np.mgrid[ij0[1]:ij1[1],ij0[0]:ij1[0]]
    xy=np.stack([xx*stride+.5,yy*stride+.5],axis=-1)
    sample=grid[yy,xx]
    camera=pose[:3,3]
    depth=(sample-camera)@pose[:3,2]
    ok=np.isfinite(sample).all(axis=-1)&(depth>.08)
    core=ok & np.all((xy>=lo+.12*extent)&(xy<=hi-.12*extent),axis=-1)
    if core.sum()<3:
        return None
    depths=depth[core]
    z=float(np.median(depths))
    mad=float(np.median(np.abs(depths-z)))
    near=np.abs(depths-z)<=max(.055,.04*z)
    if near.sum()<3 or near.mean()<.65:
        return None
    try:
        ray=np.linalg.solve(K,np.r_[.5*(lo+hi),1.])
    except np.linalg.LinAlgError:
        return None
    center=camera+pose[:3,:3]@ray*z
    ring=ok & ~np.all((xy>=lo-.06*extent)&(xy<=hi+.06*extent),axis=-1)
    plane=_surface(sample[ring],center,camera)
    clipped=min(x,y,w-u,h-v)<=max(2.,.008*min(h,w))
    pixel_m=z/min(K[0,0],K[1,1])
    # Depth scatter, grid resolution and partial boxes can all bias the inferred center.
    sigma=.035+1.48*mad+.40*stride*pixel_m+(.12 if clipped else .045)*max(extent)*pixel_m
    fid=view.get('frame_id') if frame_id is None else frame_id
    if fid is None or not np.isfinite(sigma) or not np.isfinite(center).all():
        return None
    dimensions=_dimensions(plane,rgb,box,lo,hi,camera,center,pose[:3,:3],K,stride,clipped,sigma)
    return Anchor(int(fid),tuple(center),tuple(camera),float(sigma),plane,_dominant_hue(rgb,box),
                  clipped,dimensions)


def _consistent_surface(a,b,delta):
    if a.surface is None or b.surface is None:
        return False
    na,nb=np.asarray(a.surface.normal),np.asarray(b.surface.normal)
    if float(na@nb)<math.cos(math.radians(25)):
        return False
    # Compare supports at the OBJECT position. Centroids of observed wall patches
    # slide along a wall as viewpoint changes and cannot serve as landmark origins.
    mid=(np.asarray(a.center)+b.center)*.5
    offsets=[float((mid-np.asarray(s.point))@s.normal) for s in (a.surface,b.surface)]
    if abs(offsets[0]-offsets[1])>.085:
        return False
    n=(na+nb)/np.linalg.norm(na+nb)
    normal_distance=abs(float(delta@n))
    tangent_distance=float(np.linalg.norm(delta-(delta@n)*n))
    # This distinguishes two mounting positions on the very same wall or table.
    return normal_distance<=.12 and tangent_distance<=.19


def _compare(a,b,appearance):
    if a.frame_id==b.frame_id:
        return 'unknown'
    delta=np.asarray(a.center)-b.center
    distance=np.linalg.norm(delta)
    uncertainty=math.hypot(a.sigma,b.sigma)
    if not np.isfinite(distance+uncertainty) or uncertainty>.24:
        return 'unknown'
    # A shared shape or color cannot rescue a different place on the same wall.
    if distance>max(.26,2.6*uncertainty+.055):
        return 'far'
    if (math.dist(a.camera,b.camera)<.10 or uncertainty>.18 or
            not _consistent_surface(a,b,delta)):
        return 'unknown'
    if a.hue is not None and b.hue is not None:
        if abs((a.hue-b.hue+180)%360-180)>68:
            return 'intrinsic_conflict'
    if a.dimensions is not None and b.dimensions is not None:
        x,y=np.asarray(a.dimensions),np.asarray(b.dimensions)
        side_ratio=np.maximum(x/y,y/x)
        aspect_change=(x[1]/x[0])/(y[1]/y[0])
        big_difference=(side_ratio>1.8)&(np.abs(x-y)>max(.085,1.5*uncertainty))
        if big_difference.any() or (max(aspect_change,1/aspect_change)>1.85
                and np.abs(x-y).max()>max(.085,1.5*uncertainty)):
            return 'intrinsic_conflict'
    if appearance=='conflict':
        return 'intrinsic_conflict'
    # Generic shape, absent texture and inconclusive color carry no positive vote.
    # With weak appearance require a tight, well-localized spatial repeat.
    if appearance=='strong':
        return 'match' if distance<=.19 and distance+uncertainty<.32 else 'unknown'
    return 'match' if distance<=.105 and distance+uncertainty<.195 else 'unknown'


def associate(query,memories,*,appearances=None,excluded=()):
    """Associate against all geometrically eligible owners, or leave a clue unresolved.

    An excluded owner must already be independently shown to be distinct, e.g.
    two vetted co-visible instances. A missing reference cannot prove distinction.
    """
    if query is None or not memories:
        return Decision('uncertain',None,'missing_map_evidence')
    excluded=set(excluded)
    appearances={} if appearances is None else appearances
    plausible=[]
    matched=[]
    for owner,refs in memories.items():
        if owner in excluded:
            continue
        pairs=[_compare(query,ref,appearances.get(owner,'unknown'))
               for ref in refs if ref is not None]
        if pairs and all(p=='far' for p in pairs):
            continue
        plausible.append(owner)
        if 'match' in pairs and not {'far','intrinsic_conflict'}.intersection(pairs):
            matched.append(owner)
    if len(plausible)==1 and matched==plausible:
        return Decision('same',matched[0],'same_supported_place')
    if not plausible:
        return Decision('distinct',None,'separate_map_positions_or_covisibility')
    reason='multiple_plausible_owners' if len(plausible)>1 else 'needs_another_view'
    return Decision('uncertain',None,reason)
