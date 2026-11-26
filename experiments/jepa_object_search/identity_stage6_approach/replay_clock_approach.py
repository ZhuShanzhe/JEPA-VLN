"""Branch a physical episode at its first recorded approach request, using a supplied floor map."""
import json
import math
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import quaternion
from PIL import Image, ImageDraw

from benchmark_api import Action
from confirm_view import FORWARD, pose_after
from evaluation.main.evaluator import evaluate
from proposal_worker import ProposalWorker
from replay_clock import Encounter, ROOT, STUDY, audit, rows, scene
from test_habitat_approach import SurveyGrid, floor_xy, yaw
from report_evidence import eligible_boxes
from visual_witness import compare_regions, corroborated_surroundings, describe, strong_object


BEGIN = 132


class LocalGrid:
    """Express a supplied two-dimensional navigability map in the recorded SLAM frame."""
    def __init__(self, base, scale, rotation, origin, anchor):
        self.base, self.scale, self.R = base, scale, rotation
        self.origin, self.anchor = origin, anchor
        self.res = base.res/scale
        self.raw_obstacle = base.raw_obstacle
        self.original_clear = set()

    def world_to_cell(self, xy):
        world = self.origin + self.R.T @ (np.asarray(xy)-self.anchor) * self.scale
        return self.base.world_to_cell(world)

    def in_bounds(self, cell):
        return self.base.in_bounds(cell)

    def traversable(self, cell):
        return cell in self.original_clear or self.base.traversable(cell)

    def certify_original_plan(self, pose, sequence):
        # The earlier planner already certified these segments when it emitted
        # the route. Import only those cells from the otherwise missing grid.
        # Nothing here reads the robot's later trajectory or the evaluator.
        for action in sequence:
            target=pose_after(pose,action,self.scale)
            if action==FORWARD:
                n=math.ceil(math.dist(pose[:2],target[:2])/(.5*self.res))
                for i in range(n+1):
                    xy=[a+(b-a)*i/n for a,b in zip(pose[:2],target[:2])]
                    self.original_clear.add(self.world_to_cell(xy))
            pose=target


class ApproachEncounter(Encounter):
    def __init__(self,path,episode,detector,pathfinder,mode):
        super().__init__(path,episode,detector)
        self.pathfinder = pathfinder
        self.commands = []
        self.mode=mode
        self.archived_pose_steps=0
        self.pose_log = []
        self.evidence = []
        self.last_xy = None
        self.estimated_range = []
        self.visibility = []
        self.bearings = []
        self.private_dir = Path(__file__).with_name("private_clock_branch_"+mode)
        self.private_dir.mkdir(mode=0o700,exist_ok=True)
        self.stop_reason = None
        self.agent._stationary_verdict = lambda *unused: dict(
            result="withheld", reason="fewer_than_two_shared_timestamped_anchors", selected_box=None)
        gate = self.agent._report_gate
        def record_gate(node,obs):
            accepted,reason = gate(node,obs)
            self.evidence.append(dict(step=int(obs.step_count),accepted=bool(accepted),reason=reason))
            key = (("live",int(obs.step_count)), self.agent._category(obs))
            boxes = eligible_boxes(self.detections.get(key,[]),obs.rgb.shape)
            records = []
            for fid,ref in self.agent._references(node):
                src = describe(*self.agent._trusted_pixels[(fid,tuple(ref.roi))])
                for box in boxes:
                    region = describe(obs.rgb,np.asarray(box)*obs.rgb.shape[1]/320)
                    stats = compare_regions(src,region)
                    if stats is not None:
                        records.append((strong_object(stats),corroborated_surroundings(stats)))
            self.visibility.append((len(boxes),len(records),sum(x for x,y in records),sum(x and y for x,y in records)))
            shape = obs.rgb.shape
            face = Image.fromarray(obs.rgb[...,:3]).resize((360,360))
            draw = ImageDraw.Draw(face)
            expected = math.remainder(math.atan2(node.point[1]-self.pose[1],node.point[0]-self.pose[0])-self.pose[2],math.tau)
            bearing = []
            for x,y,u,v in boxes:
                draw.rectangle((x*360/320,y*360/320,u*360/320,v*360/320),outline="lime",width=3)
                b = math.atan((159.5-(x+u)/2)/(160/math.tan(math.radians(60))))
                bearing.append(round(math.degrees(b),1))
            note = dict(step=int(obs.step_count),expected_deg=round(math.degrees(expected),1),candidate_bearings_deg=bearing,
                        boxes=len(boxes),strong=any(x for x,y in records))
            self.bearings.append(note)
            draw.text((4,5),f"{obs.step_count} | expected {note['expected_deg']} | boxes {bearing}",fill="white",stroke_width=2,stroke_fill="black")
            face.save(self.private_dir/f"{obs.step_count:03d}.jpg",quality=88)
            return accepted,reason
        self.agent._report_gate = record_gate
        self.probe_summary = []
        rank_probe=self.agent._rank_probe
        def rank(*args):
            if mode.startswith("recorded") and self.agent.now==BEGIN and not self.probe_summary:
                original, = [r for r in rows(path.parent/"agent/confirm_events.jsonl") if r["step"]==BEGIN
                             and r["event"]=="confirm_view_start" and r["frame"]==args[0]]
                out=dict(sequence=original["planned_actions"],count=original["options"],score=original["predicted_score"])
                source="original_ranker_output_saved_before_first_action"
                if mode=="recorded_inputs":
                    self.agent._frontier_grid.certify_original_plan(self.pose,out["sequence"])
            else:
                out=rank_probe(*args)
                source="new_call_frozen_ranker"
            self.probe_summary.append(dict(step=int(self.agent.now),available=bool(out),source=source,
                                           options=out["count"] if out else 0,plan_length=len(out["sequence"]) if out else 0))
            return out
        self.agent._rank_probe=rank

    def act(self,obs):
        step = int(obs.step_count)
        a = self.agent
        if step == 12:
            self.capture(step,obs)
            self.register(step)
        if step < BEGIN:
            return self.steps[step]["action"]
        if self.dispatched is not None:
            a._settle_report(obs)
            return int(Action.FINISH)
        xy = np.asarray(floor_xy(obs.gps))
        x,y,z,w = obs.extras["agent_rotation_xyzw"]
        theta = yaw(np.quaternion(w,x,y,z))
        if step==BEGIN:
            self.register(step)
            self.anchor = np.asarray(self.poses[step][:2])
            self.origin = xy.copy()
            self.rotation = self.poses[step][2] - theta
            c,s = math.cos(self.rotation), math.sin(self.rotation)
            self.R = np.array([[c,-s],[s,c]])
            base = SurveyGrid(self.pathfinder, xy, obs.gps[1])
            a._frontier_grid = LocalGrid(base,self.scale,self.R,self.origin,self.anchor)
            a.target_instance_id = self.actions[step]["target_instance_id"]
        self.pose = (*tuple(self.anchor + self.R @ (xy-self.origin)/self.scale),
                     math.remainder(theta+self.rotation,math.tau))
        if self.mode=="recorded_inputs" and step in self.poses and self.commands==[s["action"] for s in self.steps[BEGIN:step]]:
            # This is the map estimate produced at the matching prefix and
            # supplied only at that timestep, independent of scoring labels.
            self.pose=tuple(self.poses[step])
            self.archived_pose_steps+=1
        self.pose_log.append(dict(step=step,estimate=self.pose,
                             old_estimate=self.poses.get(step),
                             xyz_error_vs_old=round(float(np.linalg.norm(np.asarray(obs.gps)-self.steps[step]["position_before"])),5)))
        a._last_motion_failed = bool(self.last_xy is not None and obs.previous_action==int(Action.MOVE_FORWARD)
                                     and np.linalg.norm(xy-self.last_xy)<.05)
        self.last_xy = xy
        a.now, a._last_observation = step, obs
        a._last_feed_info = {"is_keyframe":False}
        node = a.instance_store.get(self.actions[BEGIN]["target_instance_id"])
        self.estimated_range.append(round(a._instance_distance_m(node),3))
        if step-BEGIN>=32:
            self.stop_reason = "approach_horizon"
            return int(Action.FINISH)
        action = a._pursue(obs,node)
        if action is None:
            self.stop_reason = "branch_yielded_to_exploration"
            return int(Action.FINISH)
        self.commands.append(int(action))
        if action==int(Action.TARGET_FOUND):
            self.dispatched = step
            self.stop_reason = "report_issued"
        return action


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--ranker",choices=["online","recorded_first","recorded_inputs"],default="online")
    mode=cli.parse_args().ranker
    path,_,manifest,ep = audit(STUDY)
    env,_,data = scene(json.loads(path.read_bytes()))
    detector = ProposalWorker(ROOT/"b_verification/env/bin/python",ROOT,"cuda")
    try:
        agent = ApproachEncounter(path,ep,detector,env.sim.pathfinder,mode)
        scored = evaluate(env,data,agent,max_steps=ep["max_steps_budget"],
                          threshold=manifest["arguments"]["threshold"],action_protocol=ep["action_protocol"],
                          max_steps_per_target=ep["max_steps_per_target"])
    finally:
        detector.close()
        env.close()
    match,=scored["episodes"]
    trace = match["paper"]["steps"]
    assert [r["action"] for r in trace[:BEGIN]]==[r["action"] for r in agent.steps[:BEGIN]]
    error = max(float(np.linalg.norm(np.array(q["position_before"])-p["position_before"]))
                for q,p in zip(trace[:BEGIN+1],agent.steps))
    assert error<.002 and not match["timed_out"]
    report = dict(scope="Real Habitat actions from a fixed historical prefix; the new approach controller, "
                  "real RGB detections, historical instance positions, reference, current-view identity and official scorer. "
                  "Controlled map uses the simulated navigation mesh and calibrated relative odometry; "
                  "the native depth-pair branch is set unavailable on this counterfactual path. "
                  "No supplied future observations or official goal positions, no full search continuation.",
                  start_step=BEGIN,original_prefix_error_m=error,paid_requests=0,
                  action_count=len(agent.commands),action_histogram=dict(Counter(Action(a).name for a in agent.commands)),
                  stop_reason=agent.stop_reason,estimated_start_distance_m=agent.estimated_range[0],
                  estimated_min_distance_m=min(agent.estimated_range),estimated_end_distance_m=agent.estimated_range[-1],
                  gate_verdict_histogram=dict(Counter(row["reason"] for row in agent.evidence)),
                  frames_with_local_box=sum(b>0 for b,d,s,c in agent.visibility),
                  frames_with_aligned_region=sum(d>0 for b,d,s,c in agent.visibility),
                  frames_with_strong_object=sum(s>0 for b,d,s,c in agent.visibility),
                  frames_with_strong_object_and_surroundings=sum(c>0 for b,d,s,c in agent.visibility),
                  first_strong_frame=next((s for s in agent.bearings if s["strong"]),None),
                  last_three_checks=agent.bearings[-3:],
                  estimated_source_camera_eligible=bool(list(agent.agent._remembered_views(agent.agent.instance_store.get(1),agent.scale))),
                  estimated_source_object_distance_m=round(agent.geometry(agent.agent.instance_store.get(1),13)[1],3),
                  inspection_route_source=mode,ranker_calls=agent.probe_summary,
                  original_route_clearance_imported=bool(agent.agent._frontier_grid.original_clear),
                  matching_prefix_native_pose_inputs=agent.archived_pose_steps,
                  report_step=agent.dispatched,pin_acknowledged=bool(agent.agent._confirmed_reference),
                  credited_distinct=match["found_count"],quota=match["required_target_count"],
                  incorrect=sum(e["is_false_positive"] for e in match["report_events"]),repeated=match["repeat_report_count"])
    (agent.private_dir/"checks.json").write_text(json.dumps(dict(commands=agent.commands,poses=agent.pose_log,
                       bearings=agent.bearings,events=agent.events),default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    print(json.dumps(report,allow_nan=False),flush=True)
    out=Path(__file__).with_name("clock_approach_"+mode+".json")
    if out.exists():
        assert json.loads(out.read_bytes())==report
    else:
        out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
