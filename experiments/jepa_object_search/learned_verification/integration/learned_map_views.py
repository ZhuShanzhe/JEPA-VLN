"""Short executable inspection routes on the agent's online occupancy grid."""
import math
from collections import deque

from confirm_view import (FORWARD, LEFT, RIGHT, MAX_ACTIONS, clear_move, clear_sightline,
                          pose_after, return_to_reference)


def relative_action(source, end, scale):
    dx, dy = (end[0]-source[0])*scale, (end[1]-source[1])*scale
    c, s = math.cos(source[2]), math.sin(source[2])
    return (c*dx+s*dy, s*dx-c*dy, math.remainder(source[2]-end[2], 2*math.pi))


def blocked_edge(pose, blocked, scale):
    return any(math.dist(pose[:2],where[:2])*scale<=.15 and
               abs(math.remainder(pose[2]-where[2],2*math.pi))<=math.radians(18) for where in blocked)


def map_views(grid, start, target, reference, scale, excluded, limit=.8, blocked=(), budget=MAX_ACTIONS):
    """Return one low-cost reachable route per angular sector and distance band."""
    if not grid or not scale or not math.isfinite(scale) or scale<=0:
        return []
    if not all(math.isfinite(x) for x in (*start,*target,*reference)):
        return []
    result=[]
    back=return_to_reference(grid,start,target,reference,scale,excluded,limit)
    if back and len(back[1])<=budget:
        at=start
        for action in back[1]:
            if action==FORWARD and blocked_edge(at,blocked,scale):
                break
            at=pose_after(at,action,scale)
        else:
            result.append((*back,"registered_camera"))
    origin_angle=math.atan2(start[1]-target[1],start[0]-target[0])
    queue=deque([(start,())])
    seen=set()
    slots={}
    while queue and len(seen)<3600:
        pose,route=queue.popleft()
        x,y,yaw=pose
        key=(round((x-start[0])*scale/.10),round((y-start[1])*scale/.10),
             round(math.remainder(yaw-start[2],2*math.pi)/(math.pi/6))%12)
        if key in seen:
            continue
        seen.add(key)
        distance=math.dist(pose[:2],target)*scale
        aim=abs(math.remainder(math.atan2(target[1]-y,target[0]-x)-yaw,2*math.pi))
        if (route and .33<=distance<=1.5 and aim<=math.radians(20)
                and all(math.dist(pose[:2],q)*scale>=.22 for q in excluded)
                and clear_sightline(grid,pose,target,scale)):
            sector=round(math.remainder(math.atan2(y-target[1],x-target[0])-origin_angle,2*math.pi)/(math.pi/4))%8
            band=0 if distance<=min(.7,limit) else 1
            cost=(len(route),aim,abs(distance-(.55 if not band else 1.0)))
            if (sector,band) not in slots or cost<slots[sector,band][0]:
                slots[sector,band]=(cost,pose,route)
        if len(route)>=budget:
            continue
        for a in (FORWARD,LEFT,RIGHT):
            if route and (route[-1],a) in ((LEFT,RIGHT),(RIGHT,LEFT)):
                continue
            nxt=pose_after(pose,a,scale)
            if a==FORWARD and (blocked_edge(pose,blocked,scale) or math.dist(start[:2],nxt[:2])*scale>2.25 or not clear_move(grid,pose,nxt)):
                continue
            queue.append((nxt,route+(a,)))
    existing={tuple(k[1]) for k in result}
    for _,pose,route in sorted(slots.values()):
        if route not in existing:
            result.append((pose,route,"map_probe"))
    return result
