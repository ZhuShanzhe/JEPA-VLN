"""Candidate routes around furniture, expressed in the local right/forward odometry frame."""
import math
from functools import lru_cache
import numpy as np


def turns(direction):
    delta=(int(direction)+6)%12-6
    return (2 if delta>0 else 3,)*abs(delta)


@lru_cache(maxsize=40000)
def path(plan):
    x,z,yaw=0.,0.,0.
    result=[]
    for action in plan:
        if action==1:
            x-=.25*math.sin(yaw)
            z+=.25*math.cos(yaw)
            result.append((x,z))
        else:
            yaw+=(1 if action==2 else -1)*math.pi/6
    return np.asarray(result).reshape(-1,2)


def proposals(anchor, budget):
    """Enumerate two translational legs with independent headings, then face the cue."""
    out=[]
    for heading in (0,-1,1,-2,2,-3,3,-4,4):
        for first,second in ((2,2),(2,4),(4,3),(5,5),(3,7)):
            for bend in (0,-2,2,-3,3):
                raw=turns(heading)+(1,)*first+turns(bend)+(1,)*second
                x,z=path(raw)[-1]
                facing=round(math.atan2(x-anchor[0],anchor[2]-z)/(math.pi/6))
                plan=raw+turns(facing-heading-bend)
                if len(plan)<=budget:
                    out.append((plan,heading,bend))
    return out


def axes(yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[c,s],[-s,c]])


def layout_features(patch,plan,xy,yaw,goal):
    local=path(plan)
    coords=local@axes(yaw)+xy
    prev=np.vstack([xy,coords[:-1]])
    centre=np.vstack([coords,(coords+prev)*.5])
    direction=coords-prev
    side=.36*np.column_stack([-direction[:,1],direction[:,0]])
    cells=patch.query(centre[:,0],centre[:,1])
    left,right=coords+side,coords-side
    edges=patch.query(np.r_[left[:,0],right[:,0]],np.r_[left[:,1],right[:,1]])
    sight=coords[-1]+np.arange(1,10)[:,None]/10*(goal-coords[-1])
    ray=patch.query(sight[:,0],sight[:,1])
    return np.asarray([np.mean(cells==0),np.mean(cells==2),np.mean(edges==2),
                       np.mean(ray==0),np.mean(ray==2),np.linalg.norm(coords[-1]-goal),
                       len(plan)/30,len(coords)/12],dtype=np.float32)
