"""Geometry from the native current camera and the measured left depth frame. No simulator dependency."""
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"path_feedback_20261128"))
sys.path.insert(0, str(HERE.parent/"pose_horizon_20261128"))
from path_geometry import nominal
from pose_data_contract import FEATURES

S = 192
SOURCE_H = 1.31

def options(forwards):
    f=(1,)
    plans=[f*forwards]
    for wait,width in ((0,1),(0,2),(1,2)):
        for left,right in ((2,3),(3,2)):
            l=(left,)*3; r=(right,)*3
            plans.append(f*wait+l+f*width+r+f*(forwards-wait)+r+f*width+l)
    poses=[nominal(p)[-1] for p in plans]
    assert max(np.linalg.norm(q[:4]-poses[0][:4]) for q in poses)<1e-6
    return plans

def query_depth(points,depth):
    x,y,z=points.T
    u=(S/2+x*(S/2)/np.maximum(z,.001)).astype(int)
    v=(S/2-y*(S/2)/np.maximum(z,.001)).astype(int)
    inside=(z>.05)&(u>=0)&(v>=0)&(u<S)&(v<S)
    depth_vals=np.zeros_like(z)
    depth_vals[inside]=depth[v[inside],u[inside]]
    supported=inside&np.isfinite(depth_vals)&(depth_vals>.05)
    return supported,(depth_vals+.12<z)&supported

def features(plan,h,pitch,anchor,box_info,depth):
    x,z,yaw=0.,0.,0.
    path=[]
    for a in plan:
        if a==1:
            for t in (.125,.25):
                px=x-t*math.sin(yaw); pz=z+t*math.cos(yaw)
                for y in (-.85,-.40,0.):
                    for side in (-.17,.17):
                        path.append([px+side*math.cos(yaw),y,pz+side*math.sin(yaw)])
            x-=.25*math.sin(yaw); z+=.25*math.cos(yaw)
        else:yaw+=(1 if a==2 else -1)*math.pi/6
    dh=h-SOURCE_H
    rays=np.linspace([x,dh,z],anchor,17)[1:-2]
    seen,occ=query_depth(rays,depth)
    ps,po=query_depth(np.array(path),depth) if path else (np.ones(1,bool),np.zeros(1,bool))
    dx,dy,dz=anchor-[x,dh,z]
    side=dx*math.cos(yaw)+dz*math.sin(yaw)
    fw=-dx*math.sin(yaw)+dz*math.cos(yaw)
    cp,sp=math.cos(math.radians(pitch)),math.sin(math.radians(pitch))
    oy=dy*cp-fw*sp
    oz=dy*sp+fw*cp
    f=[x,z,h,dh,math.sin(yaw),math.cos(yaw),sp,cp]+box_info+[
        np.clip(side/max(.1,oz),-5,5),np.clip(oy/max(.1,oz),-5,5),np.clip(oz,-10,10),
        seen.mean(),occ.sum()/max(1,seen.sum()),ps.mean(),(ps&~po).mean()]
    assert len(f)==len(FEATURES)
    return list(map(float,f)),bool(ps.mean()>=.65 and po.sum()==0),x,z,yaw

def turns(delta):
    delta=(delta+6)%12-6
    return (2 if delta>=0 else 3,)*abs(delta)

def endpoints(n,anchor):
    yield "old",-1,[(i,p) for i,p in enumerate(options(n))]
    vid=0
    for length,yaw in [(a,b) for a in (2,4,6) for b in (-3,-2,-1,0,1,2,3)]:
        if length==n and yaw==0:continue
        angle=yaw*math.pi/6
        x,z=-.25*length*math.sin(angle),.25*length*math.cos(angle)
        face=round(math.atan2(x-anchor[0],anchor[2]-z)/(math.pi/6))
        choices=[]
        for pid,body in enumerate(options(length)[:3]):
            p=turns(yaw)+body+turns(face-yaw)
            if len(p)<=n+16:choices.append((pid,p))
        if choices:
            yield "alt",vid,choices
            vid+=1

def map_features(plan,anchor,depth):
    end=nominal(plan)
    mov=end[end[:,4]>.5]
    rays=np.linspace([end[-1,0],0.,end[-1,1]],anchor,18)[1:-2]
    points=np.column_stack((mov[:,0],np.full(len(mov),-.60),mov[:,1]))
    last=points[-1:]+np.array([[-.15,0.,0.],[0,0.,0.],[.15,0.,0.]])
    a=math.pi/6
    def probe(p):
        px,y,pz=p.T
        cam=np.column_stack((px*math.cos(a)+pz*math.sin(a),y,-px*math.sin(a)+pz*math.cos(a)))
        ok,hit=query_depth(cam,depth)
        if not ok.any():return 0.,0.,0.
        x,y,z=cam[ok].T
        u=(S/2+x*(S/2)/z).astype(int);v=(S/2-y*(S/2)/z).astype(int)
        return float(ok.mean()),float(hit[ok].mean()),float(np.median(np.clip(depth[v,u]-z,-4.,4.)))
    p=probe(points);t=probe(rays);e=probe(last)
    return [p[0],p[1],t[0],t[1],p[2],e[0],e[1],t[2]]
