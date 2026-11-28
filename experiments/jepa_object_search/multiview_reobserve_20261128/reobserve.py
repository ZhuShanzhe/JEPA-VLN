"""Finite, RGB-D-only subroutine to reacquire a grounded cue after strict sites fail.

Pose is a map-unit (x,y,yaw); anchors, ranges, and motion distances are metric.
No action is a declaration, an arrival, or a substitute for a final observation site.
"""
import math
import numpy as np

from observation_sites import _Map, STEP, TURN, YAW_TOL


def project_original(anchor, pose, scale, hfov, shape, camera_height):
    x, y, yaw = pose[0]*scale, pose[1]*scale, pose[2]
    dx,dy=anchor["anchor_xy"][0]-x, anchor["anchor_xy"][1]-y
    along=dx*math.cos(yaw)+dy*math.sin(yaw)
    if along<=0:
        return None
    lateral=dx*math.sin(yaw)-dy*math.cos(yaw)
    h,w=shape
    focal=.5*w/math.tan(math.radians(hfov)/2)
    u=.5 + lateral*focal/(w*along)
    v=.5 - (anchor["anchor_height_m"]-camera_height)*focal/(h*along)
    return (u,v,along) if 0<=u<1 and 0<=v<1 else None


class Reobserve:
    # At most two translations (0.5 m) plus up to two corrective yaw primitives;
    # same shortest observation radius as the original strict-site generator.
    max_actions=4
    max_forwards=2
    min_standoff=.72

    def __init__(self, anchor, category):
        if anchor.get("anchor_source")!="rgbd_measured" or anchor.get("anchor_height_m") is None:
            raise ValueError("re-observation requires a measured, aligned RGB-D point")
        self.anchor=anchor
        self.category=category
        self.steps=0
        self.forwards=0
        self.awaiting=False
        self.end=None
        self.visited=set()
        self.last_pose=None
        self.last_action=None
        self.matches=0

    def finish(self,reason):
        self.end=reason
        return None

    def next_action(self, grid, pose, scale, budget):
        if self.awaiting:
            raise RuntimeError("fresh observation required before any further motion")
        if self.end:
            return None
        if budget<1 or self.steps>=self.max_actions:
            return self.finish("action_budget")
        if self.forwards>=self.max_forwards:
            return self.finish("translation_budget")
        here=(pose[0]*scale,pose[1]*scale,pose[2])
        signature=tuple(round(a,3) for a in (*here[:2],math.remainder(here[2],math.tau)))
        if signature in self.visited:
            return self.finish("repeated_pose")
        self.visited.add(signature)
        xy=self.anchor["anchor_xy"]
        dyaw=math.remainder(math.atan2(xy[1]-here[1],xy[0]-here[0])-here[2],math.tau)
        if abs(dyaw)>YAW_TOL:
            act=2 if dyaw>0 else 3
        else:
            forward=_Map(grid,scale).forward(here)
            if forward is None:
                return self.finish("no_witnessed_forward")
            remaining=math.dist(forward[:2],xy)
            if remaining<self.min_standoff or remaining>=math.dist(here[:2],xy):
                return self.finish("no_safe_distance_gain")
            act=1
        self.steps+=1
        self.forwards+=int(act==1)
        self.awaiting=True
        self.last_pose=here
        self.last_action=act
        return act

    def observe(self,grid,pose,scale,depth,boxes,collided=False,hfov=90.,camera_height=1.31):
        if not self.awaiting:
            raise RuntimeError("no outstanding re-observation action")
        self.awaiting=False
        xy=(pose[0]*scale,pose[1]*scale)
        if collided or (self.last_action==1 and math.dist(xy,self.last_pose[:2])<STEP/2):
            return self.finish("contact_or_no_progress")
        depth=np.asarray(depth).squeeze()
        if depth.ndim!=2:
            raise ValueError("expected measured depth")
        projected=project_original(self.anchor,pose,scale,hfov,depth.shape,camera_height)
        if projected is None:
            return self.finish("anchor_out_of_view")
        u,v,expected=projected
        h,w=depth.shape
        x,y=int(u*w),int(v*h)
        values=depth[max(0,y-1):min(y+2,h),max(0,x-1):min(x+2,w)]
        valid=values[np.isfinite(values)&(values>.15)]
        if not valid.size or abs(float(np.median(valid))-expected)>max(STEP,2*grid.res*scale):
            return self.finish("anchor_depth_inconsistent")
        # Only one detector box may own the projected, independently depth-checked
        # original target point. Similar overlapping or neighboring boxes lead to abstention.
        matches=[b for b in boxes if b["bbox_norm"][0]<=u<b["bbox_norm"][2]
                 and b["bbox_norm"][1]<=v<b["bbox_norm"][3] and b["category"]==self.category]
        if len(matches)!=1:
            return self.finish("ambiguous_match" if matches else "no_fresh_matching_box")
        self.matches+=1
        return matches[0]
