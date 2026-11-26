"""Evaluate the report branch at the recorded clock encounter, without a VLM call."""
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

from approach_agent import ApproachAgent
from benchmark_api import Action
from evaluation.main.evaluator import evaluate
from proposal_worker import ProposalWorker
from summarize_trial import STUDY, audit, rows
from test_chain import ReportChainTests
from verify_habitat import scene


ROOT = Path(os.environ["NAV_ROOT"])
HERE = Path(__file__).resolve().parent
FIRST, LAST = 137, 140


class Encounter:
    _benchmark_observation_api = True

    def __init__(self, path, episode, detector):
        self.steps = episode["paper"]["steps"]
        logs = path.parent / "agent"
        self.observations = episode["paper"]["agent_trace"]["observations"]
        self.actions = rows(logs / "action_trace.jsonl")
        self.decisions = {r["step"]: r for r in rows(logs / "decision_trace.jsonl")}
        self.poses = {r["step"]: r["pose"] for r in rows(logs / "confirm_events.jsonl")
                      if r.get("event") in {"confirm_view_action", "confirm_view_recheck"} and "pose" in r}
        self.stationary = {(r["step"],r["source"]):r for r in rows(path.parent / "stage4_geometry.jsonl")
                           if r["event"]=="stationary_end"}
        self.cameras = {}
        self.detections = {}
        self.registered = set()
        self.query_rows = []
        self.events = []
        self.dispatched = None
        self.scale = self.decisions[132]["agent_snapshot"]["metric_transform"]["scale"]
        self.source_pose = self.from_navigation(12)
        self.source_error = None
        self.pose = None
        self.detector = detector
        fixture = ReportChainTests()
        fixture.setUp()
        self.agent = agent = fixture.make_agent(ApproachAgent)
        # Restore the ordinary report transaction. The fixture substitutes other
        # navigation callbacks that are never invoked in this report-only replay.
        agent._geometry = self.geometry
        agent._estimated_current_pose = lambda: self.pose
        agent._metric_scale_value = lambda: self.scale
        agent._instance_distance_m = lambda n: self.scale * math.dist(n.point[:2],self.pose[:2])
        agent._probe_context = {}
        agent._geometry_log = lambda *args, **kwargs: None
        agent._note = lambda event, iid=None, **kw: self.events.append(dict(event=event, iid=iid, **kw))
        # get_frame_pose is consumed here only by MOST's 2-D navigation projection.
        # The contemporaneous source navigation pose is rounded by the source logger.
        camera = np.eye(4)
        px,py,yaw = self.source_pose
        camera[:3,3] = [px,py,0.]
        camera[:3,:3] = [[-math.sin(yaw),0.,math.cos(yaw)],[math.cos(yaw),0.,math.sin(yaw)],[0.,1.,0.]]
        agent.client.get_frame_pose = lambda fid: camera if int(fid)==13 else None
        agent.client.get_frame_image = lambda fid: ({"found":fid in self.cameras}, self.cameras.get(fid))
        agent._boxes = self.detect
        # The original online mapper's withheld result is an explicitly supplied
        # boundary input. Recreate neither a fictitious depth pair nor its score.
        agent._identity_observations = lambda *args: None
        agent._ready_native = lambda *args: None
        agent._stationary_verdict = self.stationary_result
        agent._capture_report_view = lambda: None
        agent._last_feed_info = {}

    def from_navigation(self,step):
        pose = self.decisions[step]["world_state"]["navigation"]["current_pose"]
        return pose["x_m"]/self.scale,pose["y_m"]/self.scale,math.radians(pose["yaw_deg"])

    def geometry(self,node,fid):
        if fid != 13:
            return None
        source, live = self.source_pose, self.pose
        return (math.dist(source[:2],live[:2])*self.scale, math.dist(source[:2],node.point[:2])*self.scale,
                math.dist(live[:2],node.point[:2])*self.scale, math.remainder(source[2]-live[2],math.tau))

    def detect(self,key,rgb,category):
        tag = (key,category)
        if tag not in self.detections:
            self.detections[tag] = self.detector.detect(rgb,category)
        return self.detections[tag]

    def stationary_result(self,fid,ref,obs,boxes):
        row = self.stationary[(obs.step_count,fid)]
        return dict(result=row["state"],reason=row["reason"],selected_box=row["selected_box"])

    def capture(self,step,obs):
        rgb = Image.fromarray(np.ascontiguousarray(obs.rgb[...,:3]))
        assert rgb.size == (1280,1280), rgb.size
        frame = rgb.resize((518,518),Image.Resampling.BICUBIC)
        if step == 12:
            historical = ROOT/"identity_stage4_geometry/attribution/private/clock_12.png"
            self.source_error = float(np.abs(np.asarray(frame).astype(float)-np.array(Image.open(historical))).mean())
        buf = io.BytesIO()
        frame.save(buf,format="PNG")
        self.cameras[self.actions[step]["mapping_frame_id"]] = buf.getvalue()

    def register(self,step):
        agent = self.agent
        snapshot = self.decisions[step]["agent_snapshot"]
        for entry in snapshot["instances"]:
            n = agent.instance_store.get(entry["instance_id"])
            for oid in entry["observation_ids"]:
                if oid in self.registered:
                    continue
                row = next(r for r in self.observations if r["observation_id"]==oid)
                assert row["step"] <= step
                rec = agent.instance_store.new_observation(tuple(entry["point"]),row["text"],
                    {"pixel_norm":row["pixel"],"bbox":row["bbox"]},frame_id=row["frame_id"],
                    candidate_id=row["candidate_id"],step=row["step"],pixel=row["pixel"],bbox=row["bbox"])
                assert rec.oid == oid
                if n is None:
                    n = agent.instance_store.create_instance(rec)
                    assert n.iid == entry["instance_id"]
                else:
                    agent.instance_store.attach_observation(n,rec)
                self.registered.add(oid)
            n.point = tuple(entry["point"])

    def act(self,obs):
        step = int(obs.step_count)
        agent = self.agent
        if self.dispatched is not None:
            agent._settle_report(obs)
            return int(Action.FINISH)
        if step in (12,LAST):
            self.capture(step,obs)
        if step in (12,132,139,LAST):
            self.register(step)
        if step < FIRST:
            return self.steps[step]["action"]
        if step > LAST:
            return int(Action.FINISH)
        self.pose = self.poses[step] if step in self.poses else self.from_navigation(step)
        agent.now = step
        agent._last_observation = obs
        agent._last_feed_info = dict(self.actions[step]["mapping"],frame_id=self.actions[step]["mapping_frame_id"])
        node = agent.instance_store.get(1)
        agent.target_instance_id = node.iid
        ok,reason = agent._report_gate(node,obs)
        from report_evidence import eligible_boxes
        from visual_witness import describe,compare_regions
        current = self.detections.get((("live",step),agent._category(obs)),[])
        boxes = eligible_boxes(current,obs.rgb.shape)
        candidates = []
        if agent._references(node):
            fid,ref = agent._references(node)[0]
            src = describe(*agent._trusted_pixels[(fid,tuple(ref.roi))])
            for box in boxes:
                s = compare_regions(src,describe(obs.rgb,np.array(box)*obs.rgb.shape[1]/320))
                candidates.append(None if s is None else list(map(float,s)))
        self.query_rows.append(dict(step=step,reported=bool(ok),reason=reason,
                                    estimated_object_distance_m=round(self.geometry(node,13)[2],3),
                                    local_detector_count=len(boxes),visual_descriptor_evidence=candidates,
                                    rivals=len(agent.instance_store.nodes)-1,
                                    trusted_source_count=len(agent._references(node))))
        if ok:
            assert not node.reported
            self.dispatched = step
            return agent._stage_report(node,obs,reason)
        return self.steps[step]["action"]


def main():
    path,_,manifest,episode = audit(STUDY)
    detector = ProposalWorker(ROOT/"b_verification/env/bin/python",ROOT,"cuda")
    env,_,data = scene(json.loads(path.read_bytes()))
    try:
        probe = Encounter(path,episode,detector)
        result = evaluate(env,data,probe,max_steps=episode["max_steps_budget"],
                          threshold=manifest["arguments"]["threshold"],action_protocol=episode["action_protocol"],
                          max_steps_per_target=episode["max_steps_per_target"])
    finally:
        env.close()
        detector.close()
    match, = result["episodes"]
    end = probe.dispatched if probe.dispatched is not None else LAST
    trace = match["paper"]["steps"]
    assert [r["action"] for r in trace[:end]] == [r["action"] for r in probe.steps[:end]]
    error = max(np.linalg.norm(np.asarray(a["position_before"])-b["position_before"])
                for a,b in zip(trace[:end+1],probe.steps))
    assert error < .002 and not match["timed_out"]
    report = dict(scope="Fixed historical approach, real rerendered RGB, both map instances, real DINO, "
                  "trusted-reference selection, report gate, competing instance checks and action receipt. "
                  "Frozen original mapper abstention supplied; its map point cloud is unavailable. "
                  "Historical active instance and approach motion were supplied. No autonomous search.",
                  original_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), paid_requests=0,
                  source_replay_mean_pixel_difference_vs_prior_pose=probe.source_error,
                  sensor_size=[episode["sensor_width"],episode["sensor_height"]],
                  max_prefix_position_error_m=float(error), rows=probe.query_rows,
                  credited_distinct=match["found_count"],false_reports=sum(r["is_false_positive"] for r in match["report_events"]),
                  repeats=match["repeat_report_count"],quota=match["required_target_count"],
                  acknowledged=bool(probe.agent._confirmed_reference),report_step=probe.dispatched)
    print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)
    dest=HERE/"clock_report_replay.json"
    if dest.exists():
        assert json.loads(dest.read_bytes())==report
    else:
        dest.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
