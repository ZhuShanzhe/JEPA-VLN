"""Controlled approach exercise in the recorded house with the unmodified task scorer.

Supply a remembered target from recorded RGB detections and calibrated camera
poses. Supply an idealized floor map and metric odometry to isolate locomotion.
The navigation controller never sees annotated object IDs or scoring viewpoints.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import quaternion

from approach_motion import plan_approach
from benchmark_api import Action
from confirm_view import clear_move
from evaluation.main.evaluator import _env_action, _near_target_ids, _reset_environment, evaluate
from proposal_worker import ProposalWorker
from report_evidence import eligible_boxes
from visual_witness import describe, compare_regions, strong_object, corroborated_surroundings
from verify_habitat import scene
from verify_rank import deny_ip_sockets

ROOT = Path(os.environ["NAV_ROOT"])
sys.path.insert(0, str(ROOT/"identity_stage5_confirmation"))
from summarize_trial import STUDY, audit


def floor_xy(point):
    return float(point[0]), float(-point[2])


def yaw(q):
    forward = quaternion.rotate_vectors(q, [0., 0., -1.])
    return math.atan2(-forward[2], forward[0])

def work_rgb(rgb):
    assert rgb.shape[0]==rgb.shape[1]
    return cv2.resize(rgb[...,:3],(320,320),interpolation=cv2.INTER_AREA)


def observations_for_fixture(env, item, data, episode):
    raw = _reset_environment(env, item, data)
    cameras, bearings, depth_points, starts = [], [], [], []
    reference_pitch = None
    initial_pitch = None
    private = ROOT/"identity_stage5_comparison/private"
    focal = 160/math.tan(math.radians(episode["sensor_hfov"])/2)
    for step, record in enumerate(episode["paper"]["steps"]):
        st = env.sim.get_agent_state()
        starts.append((step, np.asarray(st.position).copy(), yaw(st.rotation)))
        if step==0:
            cam=next(v for k,v in st.sensor_states.items() if "rgb" in k.lower())
            initial_pitch=math.degrees(math.asin(np.clip(quaternion.rotate_vectors(cam.rotation,[0.,0.,-1.])[1],-1,1)))
        if step in (138, 139, 140):
            cam = next(v for k,v in st.sensor_states.items() if "rgb" in k.lower())
            if step==138:
                forward=quaternion.rotate_vectors(cam.rotation,[0.,0.,-1.])
                reference_pitch=math.degrees(math.asin(np.clip(forward[1],-1,1)))
            box, = json.loads((private/f"JEPA_{step}_boxes.json").read_bytes())
            u,v=(int(round(t)) for t in ((box[0]+box[2])/2,(box[1]+box[3])/2))
            ray=quaternion.rotate_vectors(cam.rotation,[(u-159.5)/focal,(159.5-v)/focal,-1.])
            cameras.append(np.asarray(cam.position).copy())
            bearings.append(ray / np.linalg.norm(ray))
            if "depth" in raw:
                native_depth=np.asarray(raw["depth"]).squeeze()
                dh,dw=native_depth.shape
                du,dv=round((u+.5)*dw/320-.5),round((v+.5)*dh/320-.5)
                depth=native_depth[dv-1:dv+2,du-1:du+2]
                z=float(np.median(depth))
                # Habitat-Lab's DepthSensor normalizes before env.step returns.
                sensor=env.sim.sensor_suite.sensors["depth"]
                spec=sensor.config
                if spec.normalize_depth:
                    z=spec.min_depth+z*(spec.max_depth-spec.min_depth)
                depth_points.append(cam.position+ray*z)
        if step==140:
            break
        if record["action_role"]=="navigate":
            raw=env.step(_env_action(record["action"]))
    if depth_points:
        return np.median(depth_points,axis=0), starts, (reference_pitch,initial_pitch), "three visual boxes plus calibrated depth (controlled localization)"
    projections=np.eye(3)[None,:,:]-np.einsum("bi,bj->bij",bearings,bearings)
    matrix=projections.sum(0)
    assert np.linalg.cond(matrix)<1e5, "Recorded views lack translation for triangulation"
    location=np.linalg.solve(matrix,np.einsum("bij,bj->i",projections,cameras))
    assert all(np.dot(location-p,b)>0 for p,b in zip(cameras,bearings))
    return location, starts, (reference_pitch,initial_pitch), "three visual boxes plus calibrated source poses (triangulated)"


class SurveyGrid:
    res = .06

    def __init__(self, pathfinder, point, height):
        self.origin=np.array(point)-5.4
        self.walkable=np.empty((181,181),bool)
        for j,i in np.ndindex(self.walkable.shape):
            x,y=self.origin+np.array([i,j])*self.res
            self.walkable[j,i]=pathfinder.is_navigable(np.array([x,height,-y],np.float32))
        # The navmesh already includes agent radius. Approximate ray occlusion
        # separately from the stricter collision mask.
        self.raw_obstacle=cv2.erode((~self.walkable).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)

    def world_to_cell(self, xy):
        return tuple(np.rint((np.asarray(xy)-self.origin)/self.res).astype(int))

    def in_bounds(self, cell):
        return 0<=cell[0]<self.walkable.shape[1] and 0<=cell[1]<self.walkable.shape[0]

    def traversable(self, cell):
        return self.walkable[cell[1],cell[0]]


def choose_starts(starts, point3, grid):
    point=floor_xy(point3)
    level=starts[138][1][1]
    rows=[]
    for t,pos,direction in starts[:137]:
        xy=floor_xy(pos)
        if abs(pos[1]-level)>.16 or not grid.in_bounds(grid.world_to_cell(xy)):
            continue
        distance=math.dist(xy,point)
        if not 1.2<=distance<=4.0:
            continue
        fraction=1.-.65/distance
        end=tuple(x+(y-x)*fraction for x,y in zip(xy,point))
        rows.append((t,distance,not clear_move(grid,xy,end)))
    picks=[]
    for label,blocked in (("open_approach",False),("obstructed_approach",True)):
        candidates=sorted((abs(d-2.2), t) for t,d,b in rows if b==blocked)
        if candidates:
            picks.append((label,candidates[0][1]))
    return picks


class FixturePolicy:
    _benchmark_observation_api = True

    def __init__(self, prefix, grid, point, detector, control, view=None, start=None, pitch_steps=0, active=False):
        self.prefix=prefix
        self.grid=grid
        self.point=point
        self.detector=detector
        self.control=control
        self.sent=False
        self.arrivals=0
        self.blocked=[]
        self.visited=[]
        self.last=None
        self.commands=[]
        self.stop_reason=None
        self.view=view
        self.start=start
        self.pitch_steps=pitch_steps
        self.last_detection_count=0
        self.matched_evidence=[]
        self.active=active
        self.inspection=[]
        self.inspecting=False
        self.proven=False
        self.pose_on_proof=None
        if self.view is not None:
            private=ROOT/"identity_stage5_comparison/private"
            self.sources=[]
            for frame in ((138,139,140) if active else (138,)):
                reference=cv2.cvtColor(cv2.imread(str(private/f"JEPA_{frame}.png")),cv2.COLOR_BGR2RGB)
                box,=json.loads((private/f"JEPA_{frame}_boxes.json").read_bytes())
                self.sources.append(describe(work_rgb(reference),box))

    def act(self, obs):
        t=int(obs.step_count)
        if t<len(self.prefix):
            return self.prefix[t]["action"]
        if self.sent or len(self.commands)>=60:
            return int(Action.FINISH)
        pos=obs.gps
        if self.start is not None and t==0:
            assert np.linalg.norm(pos-self.start)<.005, "modified start must be realized by Habitat"
        x,y,z,w=obs.extras["agent_rotation_xyzw"]
        p=(*floor_xy(pos),yaw(np.quaternion(w,x,y,z)))
        if self.last and obs.previous_action==int(Action.MOVE_FORWARD) and math.dist(p[:2],self.last[:2])<.05:
            self.blocked.append(self.last)
        self.last=p
        if self.control:
            self.sent=True
            return int(Action.TARGET_FOUND)
        for _ in range(2):
            mode={} if self.view is None else {"observation_view": self.view}
            path=() if self.inspecting else plan_approach(self.grid,p,self.point,1.,.65,self.blocked,visited_views=self.visited,**mode)
            if path is None:
                self.stop_reason="no_reachable_view"
                return int(Action.FINISH)
            if path:
                self.commands.append(int(path[0]))
                return path[0]
            self.arrivals+=1
            if self.pitch_steps:
                direction = 1 if self.pitch_steps>0 else -1
                self.pitch_steps-=direction
                action=int(Action.LOOK_UP if direction==1 else Action.LOOK_DOWN)
                self.commands.append(action)
                return action
            rgb=np.ascontiguousarray(obs.rgb[...,:3])
            boxes=eligible_boxes(self.detector.detect(rgb,"clock"),rgb.shape)
            rgb=work_rgb(rgb)
            diagnostics=os.getenv("APPROACH_PRIVATE_CHECKS")
            if diagnostics:
                folder=Path(diagnostics)
                folder.mkdir(parents=True,exist_ok=True,mode=0o700)
                cv2.imwrite(str(folder/f"arrive_{len(self.commands)}_{self.arrivals}.jpg"),
                            cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                print("private_check",len(self.commands),self.arrivals,[list(b) for b in boxes],flush=True)
            self.last_detection_count=len(boxes)
            if self.view is None:
                supported=any(abs((b[0]+b[2])/2-rgb.shape[1]/2)<rgb.shape[1]*.25 for b in boxes)
            else:
                def match(box):
                    current=describe(rgb,box)
                    stats=[compare_regions(source,current) for source in self.sources]
                    return any(strong_object(v) and corroborated_surroundings(v) for v in stats), stats
                pairs=[match(b) for b in boxes]
                self.matched_evidence=[[None if v is None else list(v) for v in stats] for _,stats in pairs]
                supported=sum(yes for yes,_ in pairs)==1
            if self.active:
                if supported:
                    if not self.proven:
                        self.proven=True
                        self.pose_on_proof=p
                        supported=False
                    else:
                        supported=abs(math.remainder(p[2]-self.pose_on_proof[2],math.tau))>math.radians(15)
            if supported:
                self.sent=True
                self.stop_reason="detected_at_close_view"
                return int(Action.TARGET_FOUND)
            self.visited.append(p[:2])
            if self.active:
                if not self.inspecting:
                    self.inspecting=True
                    # Look toward the nearest detected region first, then across the other side.
                    middle=(boxes[0][0]+boxes[0][2])/2 if len(boxes)==1 else 160
                    toward=int(Action.TURN_RIGHT if middle>160 else Action.TURN_LEFT)
                    other=int(Action.TURN_LEFT if toward==int(Action.TURN_RIGHT) else Action.TURN_RIGHT)
                    self.inspection=[toward,other,other,toward]
                if self.inspection:
                    a=self.inspection.pop(0)
                    self.commands.append(a)
                    return a
            if self.view:
                self.stop_reason="remembered_view_not_visually_verified"
                return int(Action.FINISH)
        self.stop_reason="no_fresh_view"
        return int(Action.FINISH)


def outside_starts(env,item,threshold,ref_xyz,grid):
    # Select only on the INITIAL scoring label and static topology, before
    # running either arm. The selected pair receives no valid goal viewpoints.
    center=np.array(floor_xy(ref_xyz))
    selected={}
    for r in (2.0,2.75,3.5,4.0):
        for angle in np.linspace(0,2*math.pi,48,endpoint=False):
            xy=center+r*np.array([math.cos(angle),math.sin(angle)])
            pos=np.array([xy[0],ref_xyz[1],-xy[1]],dtype=np.float32)
            if not env.sim.pathfinder.is_navigable(pos):
                continue
            pos=np.asarray(env.sim.pathfinder.snap_point(pos),dtype=np.float32)
            if np.linalg.norm(pos-np.array([xy[0],ref_xyz[1],-xy[1]]))>.18:
                continue
            if _near_target_ids(pos,item.target_object_ids,item.target_positions,threshold,
                                                      env.sim.pathfinder,item.view_points):
                continue
            # Exclude disconnected fixture starts, not locally inconvenient detours.
            import habitat_sim
            sp=habitat_sim.ShortestPath()
            sp.requested_start=pos
            sp.requested_end=ref_xyz
            if env.sim.pathfinder.find_path(sp) and sp.geodesic_distance<=r+2.5:
                delta=sp.geodesic_distance-math.dist(pos,ref_xyz)
                if delta<.025:
                    label="open_approach"
                elif delta>=.20:
                    label="obstructed_approach"
                else:
                    continue
                if label not in selected:
                    selected[label]=(pos.copy(),float(delta))
            if len(selected)==2:
                return selected
    return selected


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--suite",choices=("object_standoff","return_to_view","return_active"),default="object_standoff")
    args=parser.parse_args()
    deny_ip_sockets()
    file,_,manifest,episode=audit(STUDY)
    env,item,data=scene(json.loads(file.read_bytes()))
    detector=ProposalWorker(ROOT/"b_verification/env/bin/python",ROOT,"cuda")
    try:
        point,starts,(reference_pitch,initial_pitch),method=observations_for_fixture(env,item,data,episode)
        grid=SurveyGrid(env.sim.pathfinder,floor_xy(point),starts[138][1][1])
        results=[]
        _,reference_pos,reference_heading=starts[138]
        view=(*floor_xy(reference_pos),reference_heading)
        distance=math.dist(view[:2],floor_xy(point))
        threshold=manifest["arguments"]["threshold"]
        if args.suite.startswith("return_"):
            outside=outside_starts(env,item,threshold,reference_pos,grid)
            cases=[(label,0,pos,delta) for label,(pos,delta) in outside.items()]
        else:
            cases=[(label,t,None,None) for label,t in choose_starts(starts,point,grid)]
        assert cases
        for label,t,pos,delta in cases:
            if pos is not None:
                target=deepcopy(item)
                target.start_position=pos.tolist()
                target.start_rotation=[0.,0.,0.,1.]
                data.episodes=[target]
            for control in (True,False):
                nav=FixturePolicy(episode["paper"]["steps"][:t],grid,floor_xy(point),detector,control,
                                  view if pos is not None else None, pos,
                                  round((reference_pitch-initial_pitch)/15.) if pos is not None else 0,
                                  args.suite=="return_active")
                doc=evaluate(env,data,nav,max_steps=300,threshold=manifest["arguments"]["threshold"],
                             action_protocol=episode["action_protocol"],max_steps_per_target=episode["max_steps_per_target"])
                ep,=doc["episodes"]
                assert not ep["timed_out"]
                final_position=np.asarray(ep["paper"]["steps"][-1]["position_before"])
                final_eligible=_near_target_ids(final_position,item.target_object_ids,item.target_positions,
                                                 threshold,env.sim.pathfinder,item.view_points)
                results.append(dict(condition=label,immediate_report_control=control,starting_step=t,
                                    geodesic_excess_over_straight_path_m=delta,
                                    report_actions=len(ep["report_events"]),credited_distinct=ep["found_count"],
                                    wrong_reports=sum(not e["near_targets"] for e in ep["report_events"]),
                                    motion_actions=len(nav.commands),forward_steps=nav.commands.count(1),
                                    blocked_forwards=len(nav.blocked),arrival_checks=nav.arrivals,
                                    detections_on_last_check=nav.last_detection_count,
                                    terminal_inside_report_range=bool(final_eligible),
                                    single_view_matching_statistics=nav.matched_evidence,
                                    stop_reason=nav.stop_reason))
                print("approach_fixture", results[-1],flush=True)
        summary=dict(task="motion module test; known return visit in one HM3D clock episode",suite=args.suite,source=method,
                     stored_view_object_distance_m=distance,stored_camera_pitch_degrees=reference_pitch,
                     initial_camera_pitch_degrees=initial_pitch,
                     map="idealized floor occupancy and calibrated odometry supplied to both arms",
                     start_selection=("first reachable out-of-scoring-range start by fixed radius, azimuth and direct path topology; judge used only in fixture construction"
                                      if args.suite.startswith("return_") else "closest archived pose to 2.2 m visual-target distance per straight-path condition; score ignored"),
                     arrival_perception=("unique RGB detection confirmed across two headings by prior object-and-surrounding landmarks" if args.suite=="return_active" else
                                         "unique local RGB detection matching object-and-surrounding landmarks in remembered frame" if args.suite=="return_to_view" else
                                         "frozen class detector in central half of RGB at close estimated range; identity supplied by fixture"),
                     reference="earlier imagery from a separate traversal; no benchmark viewpoint, semantic ID, or score fed to policy",
                     baseline="immediate report from same location with same memory; not original MOST-Nav",
                     paid_calls=0,results=results)
        args.out.write_text(json.dumps(summary,indent=2)+"\n")
    finally:
        detector.close()
        env.close()

if __name__=="__main__":
    main()
