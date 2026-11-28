"""Oracle-only isolation of a target at each recorded camera pose; never used for navigation."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np
import habitat_sim as hs
import magnum as mn

HERE=Path(__file__).resolve().parent
VIEWS=HERE.parent/"new_view_20261129"
LOGS=HERE.parent/"reactive_view_20261129"
REFIT=HERE.parent/"refined_confirmation_20261129"
sys.path.insert(0,str(VIEWS))
from sample_views import simulator,inventory,ASSETS,camera,SOURCE_H,ACTION

H=SOURCE_H
FAR=np.array([128.,0.,128.])
CUT=369


def centered(sim, eye, center, sid):
    d=center-eye
    theta=math.atan2(-d[0],-d[2])
    pitch=math.degrees(math.atan2(d[1],np.linalg.norm(d[[0,2]])))
    obs=camera(sim,eye-[0,H,0],theta,H,pitch)
    return int((obs["ids"]==sid).sum())


def yaw_aim(sim,eye,center,sid):
    d=center-eye
    yaw=math.atan2(-d[0],-d[2])
    return int((camera(sim,eye-[0,H,0],yaw,H,0)["ids"]==sid).sum())


def probe(house):
    sim=simulator(house)
    try:
        inventory(sim,ASSETS)
        # Triple the canvas at the original focal length, with an identically aligned centre crop.
        wide=hs.CameraSensorSpec()
        wide.uuid="wide_instances";wide.sensor_type=hs.SensorType.SEMANTIC
        wide.resolution=[576,576];wide.hfov=mn.Deg(math.degrees(2*math.atan(3)))
        wide.position=mn.Vector3(0,H,0)
        sim.add_sensor(wide)
        ag=sim.get_agent(0)
        objs=sim.get_rigid_object_manager().get_objects_by_handle_substring("")
        targets={o.semantic_id:o for o in objs.values()}
        rows=[]
        for p in sorted((REFIT/"private_inputs"/house).glob("*.npz")):
            r=json.loads((LOGS/"private_roll"/house/(p.stem+".json")).read_text())
            with np.load(VIEWS/"private_render_blind60"/("blind_"+house)/(p.stem+"_sensor.npz")) as s:
                start,yaw,sid=s["position"],float(s["yaw"]),int(s["oid"])
            obj=targets[sid]
            begin=obj.transformation
            mt=obj.motion_type
            center=np.asarray(begin.transform_point(obj.aabb.center()),float)
            logs=r["events"];old=r["new_confirmation"]
            stop=json.loads((REFIT/"private_paired/full"/house/p.stem/"decision.json").read_text())["actions_used"]
            never_before=old is None and max(e["px"] for e in logs)<CUT
            result_at_old="wrong" if old is not None and not logs[old]["correct"] else (
                "right" if old is not None else "never" if never_before else "missed")
            # Acquire all real actions (including realized floor slopes/collisions), then freeze poses.
            poses=[];native=[]
            obs=camera(sim,start,yaw,H,0)
            for t,e in enumerate(logs):
                if t:
                    ag.act(ACTION[e["action"]]);obs=sim.get_sensor_observations()
                mask=obs["ids"]==sid
                assert int(mask.sum())==e["px"],(p.stem,t,int(mask.sum()),e["px"])
                poses.append((ag.state.position.copy(),2*math.atan2(ag.state.rotation.y,ag.state.rotation.w)))
                native.append(mask.copy())
            # In the intact furnished scene: revisit each actual optical centre, with yaw then yaw+pitch targeting the object's bounding-box centre.
            room_yaw=[];room_aim=[]
            am={}
            for base,theta in poses:
                key=base.tobytes()
                if key not in am:
                    eye=base+[0,H,0]
                    am[key]=(yaw_aim(sim,eye,center,sid),centered(sim,eye,center,sid))
                y,a=am[key];room_yaw.append(y);room_aim.append(a)
            obj.motion_type=hs.physics.MotionType.KINEMATIC
            obj.translation=mn.Vector3(*(np.asarray(begin.translation)+FAR))
            prime=[];wide_areas=[];superset_loss=[];crop_diff=[]
            for t,(base,theta) in enumerate(poses):
                obs=camera(sim,base+FAR,theta,H,0)
                isolate=obs["ids"]==sid
                full=obs["wide_instances"]==sid
                prime.append(int(isolate.sum()))
                wide_areas.append(int(full.sum()))
                superset_loss.append(int((native[t]&~isolate).sum()))
                crop_diff.append(int(np.logical_xor(full[192:384,192:384],isolate).sum()))
            # Separate projected footprint from range/azimuth at the best actually seen native frame.
            best=int(np.argmax([m.sum() for m in native]))
            base,theta=poses[best]
            c=center+FAR
            eye=base+[0,H,0]+FAR
            v=eye-c
            distance=float(np.linalg.norm(v));horizontal=np.linalg.norm(v[[0,2]])
            area_center=centered(sim,eye,c,sid)
            areas=[area_center]
            for k in range(1,8):
                rad=math.atan2(v[2],v[0])+k*math.pi/4
                test=c+[horizontal*math.cos(rad),v[1],horizontal*math.sin(rad)]
                areas.append(centered(sim,test,c,sid))
            close=centered(sim,c+.65*v,c,sid)
            same_axis_close=int((camera(sim,c+.65*v-[0,H,0],theta,H,0)["ids"]==sid).sum())
            # Restore the object before reproducing the next request's physical path.
            obj.transformation=begin
            obj.motion_type=mt
            rows.append(dict(old=result_at_old,stop=stop,native=[int(m.sum()) for m in native],isolated=prime,
                             expanded=wide_areas,lost=superset_loss,crop_diff=crop_diff,peak_step=best,
                             target_center_distance=distance,centered=area_center,azimuth=areas,closer=close,
                             near_same_optical_axis=same_axis_close,room_yaw=room_yaw,room_aim=room_aim))
    finally:
        sim.close()
    private=HERE/"private_occlusion";private.mkdir(exist_ok=True,mode=0o700)
    (private/(house+".json")).write_text(json.dumps(rows))


def summarize(rows):
    result=Counter();fraction=[];distance=[]
    for r in rows:
        p=r["peak_step"];true=np.array(r["native"]);isol=np.array(r["isolated"]);wide=np.array(r["expanded"])
        result["groups"]+=1
        result["true_clear_on_any_view"]+=int(true.max()>=CUT)
        result["isolated_clear_on_any_view"]+=int(isol.max()>=CUT)
        result["newly_clear_on_same_visited_view_without_other_surfaces"]+=int(true.max()<CUT<=isol.max())
        result["expanded_canvas_only_any_view"]+=int(isol.max()<CUT<=wide.max())
        result["native_peak_unchanged_within_two_pixels"]+=int(abs(true[p]-isol[p])<=2)
        ratio=float(true[p]/max(1,isol[p]))
        result["native_peak_more_than_half_occluded"]+=int(ratio<.5)
        result["native_peak_at_least_quarter_occluded"]+=int(ratio<=.75)
        result["native_peak_10pct_or_less_occluded"]+=int(ratio>=.9)
        result["native_peak_below_cut_becomes_clear_after_isolation"]+=int(true[p]<CUT<=isol[p])
        result["native_peak_clipped_at_least_10pct_using_expanded_canvas"]+=int(wide[p]>0 and isol[p]/wide[p]<=.9)
        order=[true[p],isol[p],r["centered"],max(r["azimuth"]),r["closer"]]
        labels=["visible","same_pose_remove_occluders","aim_at_center_only","different_bearing_same_range_and_elevation",
                "closer_on_initial_bearing","below_even_in_these_probes"]
        result["peak_first_factor_"+labels[next((i for i,x in enumerate(order) if x>=CUT),5)]]+=1
        result["peak_still_small_on_same_bearing_then_clear_35pct_closer"]+=int(r["centered"]<CUT<=r["closer"])
        result["peak_isolated_below_cut_clear_35pct_closer_same_optical_axis"]+=int(isol[p]<CUT<=r["near_same_optical_axis"])
        manual=np.column_stack((true,r["room_yaw"],r["room_aim"]))
        result["any_original_below_then_clear_after_yaw_only_intact_scene"]+=int(true.max()<CUT and max(r["room_yaw"])>=CUT)
        result["any_original_below_then_clear_after_yaw_and_pitch_intact_scene"]+=int(true.max()<CUT and max(r["room_aim"])>=CUT)
        result["any_needs_pitch_beyond_attempted_yaw_intact_scene"]+=int(manual[:,:2].max()<CUT and max(r["room_aim"])>=CUT)
        result["any_either_aim_mode_can_recover_in_intact_scene"]+=int(true.max()<CUT and manual[:,1:].max()>=CUT)
        result["peak_would_be_clear_by_same_pose_isolation_or_35pct_closer_original_axis"]+=int(true[p]<CUT and max(isol[p],r["near_same_optical_axis"])>=CUT)
        fraction.append(ratio);distance.append(r["target_center_distance"])
    return dict(counts=dict(result),median_visible_fraction_at_native_peak=round(float(np.median(fraction)),3),
                median_distance_to_object_box_center_at_peak_m=round(float(np.median(distance)),3))


def main():
    p=argparse.ArgumentParser();p.add_argument("--house-index",type=int);args=p.parse_args()
    houses=json.loads((VIEWS/"view_design.json").read_text())["cohorts"]["blind"]
    if args.house_index is not None:
        probe(houses[args.house_index]);return
    rows=[r for h in houses for r in json.loads((HERE/"private_occlusion"/(h+".json")).read_text())]
    assert len(rows)==94
    report=dict(protocol="Oracle diagnostic in cloned physics. Target plus camera translated 128m on two axes into empty space, same relative metric pose. The wide camera uses triple the field canvas at unchanged focal length, so its middle third has the original projection. Real trajectory re-render checked against every original frame. Best frame chosen by actual visible pixels, breaking ties at first encounter; all original no-stop trajectories are included. Angles/range tests are ideal single-object views and need not be navigable.",
                native_pixels_missing_from_isolated=sum(sum(r["lost"]) for r in rows),
                max_native_pixels_missing_single_frame=max(max(r["lost"]) for r in rows),
                wide_crop_max_difference_pixels=max(max(r["crop_diff"]) for r in rows),
                actions_and_initial_frames=sum(len(r["native"]) for r in rows),
                all=summarize(rows),old33_never_clear=summarize([r for r in rows if r["old"]=="never"]),
                old29_seen_unreported=summarize([r for r in rows if r["old"]=="missed"]))
    (HERE/"occlusion_attribution.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)


if __name__=="__main__":main()
