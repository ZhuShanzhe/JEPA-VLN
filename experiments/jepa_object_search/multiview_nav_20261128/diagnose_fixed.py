"""Offline filter audit on the previous frozen handoffs. No evaluation labels or changes to policy."""
from collections import Counter
import json
import math
import numpy as np
from unittest.mock import patch

import replay_handoffs as replay
from observation_sites import _Map, _route, _depth_hit


def depth_detail(choice, depth):
    l,t,r,b = choice["bbox_norm"]
    u,v = (l+r)/2, (t+b)/2
    z = np.asarray(depth).squeeze()
    h,w = z.shape
    f = w/2
    cx,cy = u*w-.5,v*h-.5
    rad = max(1,min(4,round(min(w,h)*.012)))
    ys,xs = np.mgrid[max(0,round(cy)-rad):min(h,round(cy)+rad+1),
                     max(0,round(cx)-rad):min(w,round(cx)+rad+1)]
    small = z[ys,xs]
    inbox = (xs+.5>=l*w)&(xs+.5<r*w)&(ys+.5>=t*h)&(ys+.5<b*h)
    finite = inbox & np.isfinite(small) & (small>.15)
    along=small
    lateral=(xs+.5-w/2)*small/f
    range_m=np.hypot(along,lateral)
    elevation=1.31+(h/2.-ys-.5)*small/f
    close=finite & (range_m<=5.)
    vertical=finite & (.13<=elevation) & (elevation<=2.9)
    def span(array):
        a=array[finite]
        return [round(float(x),3) for x in (a.min(),np.median(a),a.max())] if a.size else None
    return dict(window_pixels=int(finite.size),min_valid_pixels=max(2,math.ceil(finite.size*.35)),
                inside_pixels=int(inbox.sum()),finite_depth_pixels=int(finite.sum()),
                optical_depth_min_median_max_m=span(small),horizontal_range_min_median_max_m=span(range_m),
                estimated_height_min_median_max_m=span(elevation),range_ok_pixels=int(close.sum()),
                height_ok_pixels=int(vertical.sum()),both_ok_pixels=int((vertical&close).sum()))


def site_detail(grid, pose, scale, anchor, budget):
    online=_Map(grid,scale)
    cur=(pose[0]*scale,pose[1]*scale,pose[2])
    counts=Counter()
    details=[]
    for radius,offset in ((.95,0.),(.95,32.),(.95,-32.),(1.4,0.),(.72,0.),(1.20,55.),(1.20,-55.)):
        bearing=math.atan2(cur[1]-anchor[1],cur[0]-anchor[0])+math.radians(offset)
        point=(anchor[0]+radius*math.cos(bearing),anchor[1]+radius*math.sin(bearing))
        yaw=math.atan2(anchor[1]-point[1],anchor[0]-point[0])
        cell=online.cell(point)
        if math.dist(cur[:2],point)<.34:
            why="within_existing_camera_exclusion"
        elif not online.observed(cell):
            why="camera_cell_unobserved"
        elif not online.free(point):
            why="camera_cell_untraversable"
        elif not online.clear_sightline(point,anchor):
            why="sightline_uncertified"
        else:
            route=_route(online,cur,point,yaw,budget,False)
            if not route:
                why="no_certified_direct_route"
            elif 1 not in route:
                why="rotation_only"
            else:
                why="accepted"
        entry=dict(stand_off_m=radius,offset_deg=offset,first_rejection=why)
        if why=="no_certified_direct_route":
            direct42=_route(online,cur,point,yaw,42,False)
            detour24=_route(online,cur,point,yaw,budget,True)
            entry.update(direct42=bool(direct42),detour24=bool(detour24))
        details.append(entry)
        counts[why]+=1
    return dict(first_rejections=dict(counts), candidates=details)


def main():
    out=[]
    original=replay.located_cue
    def measured(choice,views,grid,pose,scale,geometry,hfov):
        anchors,source,yaw=original(choice,views,grid,pose,scale,geometry,hfov)
        row=dict(goal=choice["category"],camera=("right","center","left")[choice["view_index"]],
                 depth=depth_detail(choice,geometry[choice["view_index"]][1]),projection=bool(anchors))
        if anchors:
            # The exact remaining budget is supplied separately by replay.check_case.
            row["pending"]=(grid,pose,scale,anchors[0]["anchor_xy"])
        out.append(row)
        return anchors,source,yaw
    original_sites=replay.propose_sites
    def inspected(grid,pose,scale,anchors,budget):
        sites=original_sites(grid,pose,scale,anchors,budget)
        if "pending" in out[-1]:
            args=out[-1].pop("pending")
            out[-1]["viewpoint_generation"]=site_detail(*args,budget)
        out[-1]["site_count"]=len(sites)
        return sites
    ch,proc=replay.start_worker(replay.ROOT,replay.HERE/"private_fixed/diagnostic_worker.log",cpu=True)
    try:
        with patch.object(replay,"located_cue",measured), patch.object(replay,"propose_sites",inspected):
            cases=[r for i,s in enumerate(replay.read(replay.FAMILY/"panel.json")) for r in replay.check_case(i,s,ch)]
    finally:
        ch.close()
        proc.wait(timeout=45)
    assert len(cases)==4 and len(out)==2
    result=dict(source_cases=len(cases),selected=len(out),new_paid_calls=0,diagnostics=out)
    (replay.HERE/"fixed_filters.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
