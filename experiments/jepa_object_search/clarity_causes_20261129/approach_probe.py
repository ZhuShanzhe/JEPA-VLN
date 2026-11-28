"""Matched physical closer-view probes from the best frames of the 33 unresolved local requests."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PREVIOUS = PROJECT / "reactive_view_20261129"
SOURCES = PROJECT / "new_view_20261129"
sys.path.insert(0, str(SOURCES))
from sample_views import simulator, inventory, camera, ASSETS, ACTION, SOURCE_H
from view_geometry import turns


def image_info(sim, oid):
    obs = sim.get_sensor_observations()
    mask = obs["ids"] == oid
    yy, xx = np.where(mask)
    depths = obs["depth"].squeeze()[mask]
    return len(yy), xx, depths


def rotate(ag, steps):
    for a in turns(int(steps)):
        ag.act(ACTION[a])


def angle(ag):
    q = ag.state.rotation
    return 2. * math.atan2(q.y, q.w)


def world_points(sim,ag,oid):
    obs=sim.get_sensor_observations()
    y,x=np.where(obs["ids"]==oid)
    depth=obs["depth"].squeeze()[y,x]
    z=depth;right=(x+.5-96)*z/96;up=(96-y-.5)*z/96
    a=angle(ag);c,s=math.cos(a),math.sin(a)
    # Subtract a common fixed sensor-height origin, valid for a fixed-height horizontal camera.
    return np.column_stack([right*c-z*s,up,-right*s-z*c])+ag.state.position


def lost_projection(sim,ag,oid,points):
    if not len(points):return None
    dx,up,dz=(points-ag.state.position).T
    a=angle(ag);c,s=math.cos(a),math.sin(a)
    z=-dx*s-dz*c;right=dx*c-dz*s
    u=np.floor(96+96*right/np.maximum(z,.001)).astype(int)
    v=np.floor(96-96*up/np.maximum(z,.001)).astype(int)
    inside=(z>.02)&(u>=0)&(u<192)&(v>=0)&(v<192)
    obs=sim.get_sensor_observations()
    dist=obs["depth"].squeeze()[v[inside],u[inside]]
    hit=obs["ids"][v[inside],u[inside]]
    nearer_other=(hit!=oid)&(dist>.02)&(dist<z[inside]-(.06+.02*z[inside]))
    return dict(source_points=len(points),outside=int((~inside).sum()),
                outside_vertical=int(((z>.02)&((v<0)|(v>=192))).sum()),
                outside_horizontal=int(((z>.02)&((u<0)|(u>=192))).sum()),
                other_in_front=int(nearer_other.sum()),
                still_target=int((hit==oid).sum()))


def remaining(r):
    return r["budget"] - r["peak_idx"]


def probe(sim,ag,root,row):
    events=row["events"]
    k=int(np.argmax([s["px"] for s in events]))
    with np.load(root/(str(row["source"])+"_sensor.npz")) as frame:
        start,yaw,oid=frame["position"],float(frame["yaw"]),int(frame["oid"])
    camera(sim,start,yaw,SOURCE_H,0.)
    for e in events[1:k+1]:
        ag.act(ACTION[e["action"]])
    original=ag.get_state()
    initial,xx,dep=image_info(sim,oid)
    assert 0<initial==events[k]["px"]<369 and np.isfinite(dep).all()
    # Offline oracle bearing to the originally visible target surface, without changing the camera.
    lateral=(xx+.5-96)/96*dep
    x,z=float(np.median(lateral)),float(np.median(dep))
    yaw=angle(ag)
    anchor=(original.position+np.array([x*math.cos(yaw)-z*math.sin(yaw),0,
                                      -x*math.sin(yaw)-z*math.cos(yaw)]))[[0,2]]
    aim=int(round(-math.atan2(x,z)/(math.pi/6)))
    def distance():
        return float(np.linalg.norm(anchor-ag.state.position[[0,2]]))
    start_distance=distance()
    rotate(ag,aim)
    turn_px=image_info(sim,oid)[0]
    points=world_points(sim,ag,oid)
    samples=[]
    for offset in (0,-1,1,-2,2,-3,3):
        ag.set_state(original,reset_sensors=False,infer_sensor_states=False)
        move_bearing=aim+offset
        rotate(ag, move_bearing)
        for i in range(1,5):
            p=ag.state.position.copy()
            hit=ag.act(ACTION[1])
            full=not hit and np.linalg.norm((ag.state.position-p)[[0,2]])>=.23
            state=ag.get_state()
            gap=distance()
            # Measure once at the original orientation, once at the exact heading of the turn-only comparator.
            rotate(ag, -move_bearing)
            px_original=image_info(sim,oid)[0]
            assert np.linalg.norm(ag.state.position-state.position)<1e-5
            rotate(ag, aim)
            px_aim=image_info(sim,oid)[0]
            projected=lost_projection(sim,ag,oid,points) if offset==0 else None
            ag.set_state(state,reset_sensors=False,infer_sensor_states=False)
            sample=dict(offset=offset,i=i,full_step=bool(full),closer=bool(start_distance-gap>.05),
                        gap=gap,px_orig=px_original,px_aim=px_aim,
                        cost_orig=len(turns(move_bearing))+i+len(turns(-move_bearing)),
                        cost_aim=len(turns(move_bearing))+i+len(turns(-offset)),projected=projected)
            samples.append(sample)
            if not full:break
    return dict(peak=initial,peak_idx=k,budget=row["budget"],distance=start_distance,
                turn_only=turn_px,turn_cost=len(turns(aim)),samples=samples)


def run(index):
    path=HERE/"private_approach"
    path.mkdir(mode=0o700,exist_ok=True)
    houses=json.loads((SOURCES/"view_design.json").read_text())["cohorts"]["blind"]
    h=houses[index]; folder=SOURCES/"private_render_blind60"/("blind_"+h)
    sim=simulator(h);out=[]
    try:
        inventory(sim,ASSETS);ag=sim.get_agent(0)
        for f in sorted((PREVIOUS/"private_roll"/h).glob("*.json")):
            r=json.loads(f.read_text())
            if r["focus"] and r["new_confirmation"] is None and max(e["px"] for e in r["events"])<369:
                out.append(probe(sim,ag,folder,r))
    finally:sim.close()
    (path/(str(index)+".json")).write_text(json.dumps(out))


def summarize():
    rows=sum([json.loads((HERE/"private_approach"/(str(i)+".json")).read_text()) for i in range(3)],[])
    assert len(rows)==33
    total=Counter(groups=len(rows)); steps={}
    for row in rows:
        for name,keep,head in (("direct",lambda s:s["offset"]==0,"aim"),("side_or_direct",lambda s:True,"orig")):
            options=[s for s in row["samples"] if keep(s) and s["full_step"] and s["closer"]]
            good=[s for s in options if s["px_"+head]>=369]
            within=[s for s in good if s["cost_"+head]<=remaining(row)]
            total[name+"_possible_closer"]+=int(bool(options))
            total[name+"_crossed_threshold"]+=int(bool(good))
            total[name+"_crossed_within_remaining_budget"]+=int(bool(within))
            before=row["turn_only"] if head=="aim" else row["peak"]
            total[name+"_rescued_with_heading_matched"]+=int(before<369 and bool(good))
            total[name+"_rescued_heading_matched_within_remaining_budget"]+=int(before<369 and bool(within))
        direct=[s for s in row["samples"] if s["offset"]==0]
        total["direct_first_step_blocked"]+=int(not direct[0]["full_step"])
        total["direct_any_blocked_before_one_m"]+=int(any(not s["full_step"] for s in direct))
        total["turn_only_crossed"]+=int(row["turn_only"]>=369)
        total["turn_only_crossed_within_remaining_budget"]+=int(row["turn_only"]>=369 and row["turn_cost"]<=remaining(row))
    for i in (1,2,3,4):
        items=[(r,s) for r in rows for s in r["samples"] if s["offset"]==0 and s["i"]==i and s["closer"] and s["full_step"]]
        if not items:continue
        steps[str(i*.25)]={"reached_and_closer":len(items),
          "median_target_pixels_before_same_heading":float(np.median([r["turn_only"] for r,s in items])),
          "median_target_pixels_after":float(np.median([s["px_aim"] for r,s in items])),
          "median_paired_pixel_change":float(np.median([s["px_aim"]-r["turn_only"] for r,s in items])),
          "median_paired_range_change_m":float(np.median([s["gap"]-r["distance"] for r,s in items])),
          "pixel_increases":sum(s["px_aim"]>r["turn_only"] for r,s in items),
          "crossed_from_below":sum(r["turn_only"]<369<=s["px_aim"] for r,s in items)}
    decreasing=[s["projected"] for r in rows for s in r["samples"] if s["i"]==1 and s["offset"]==0 and s["full_step"] and s["closer"] and s["px_aim"]<r["turn_only"]]
    geometry={"decreased_after_first_accessible_direct_step":len(decreasing),
        "majority_source_pixels_offscreen":sum(d["outside"]>.5*d["source_points"] for d in decreasing),
        "majority_source_pixels_past_top_or_bottom":sum(d["outside_vertical"]>.5*d["source_points"] for d in decreasing),
        "majority_source_pixels_past_left_or_right":sum(d["outside_horizontal"]>.5*d["source_points"] for d in decreasing),
        "majority_source_pixels_behind_nearer_non_target":sum(d["other_in_front"]>.5*d["source_points"] for d in decreasing),
        "majority_source_pixels_still_project_to_same_instance":sum(d["still_target"]>.5*d["source_points"] for d in decreasing),
        "notes":"Reproject actually visible instance depth samples from the turn-only frame into the same heading 0.25m closer. Count other-instance depth at least 0.06m + 2% depth nearer as physical occlusion; mixed and rasterization losses can occur."}
    report={"total":dict(total),"distance_steps_m_direct":steps,"first_step_pixel_loss_diagnosis":geometry,
      "median_planar_distance_to_visible_surface_m_at_peak":float(np.median([r["distance"] for r in rows])),
      "median_peak_pixels":float(np.median([r["peak"] for r in rows])),
      "protocol":"Earliest best target-pixel frame in each of the frozen 33 never-clear, unreported requests. Center bearing estimated offline from the actual instance pixels and depth, rounded to the existing 30-degree action. Compare in-place turn to the same camera heading after actual forward moves of 0.25–1m. Test six extra bearings with original camera heading restored at the endpoint. Count only completed collision-free movements that shorten horizontal range to the original visible target surface by >0.05m. Resource counts use the minimum physical turns to the corresponding heading including return, with original request budget minus prefix to peak. Diagnostic uses oracle target region and is not a navigation score; changing distance can change occlusion and target surface angles."}
    (HERE/"approach_probe.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--index",type=int);i=ap.parse_args().index
    if i is None:summarize()
    else:run(i)
