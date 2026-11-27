"""Evaluate the frozen transferred arrival gate after Habitat controls a return to a known view."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import quaternion
from approach_agent import ApproachAgent
from approach_motion import plan_approach
from benchmark_api import Action
from evaluation.main.evaluator import evaluate
from proposal_worker import ProposalWorker
from test_chain import ReportChainTests
from test_habitat_approach import ROOT, STUDY, SurveyGrid, audit, floor_xy, observations_for_fixture, outside_starts, scene, yaw
from transfer_agent import ArrivalTransferAgent
from trusted_reference import ReferenceEvidence
from verify_rank import deny_ip_sockets


class ReturnWithIntent:
    _benchmark_observation_api = True

    def __init__(self, grid, point, view, detector, control, turn):
        fixture=ReportChainTests()
        fixture.setUp()
        self.agent=a=fixture.make_agent(ArrivalTransferAgent)
        # Supply the same three pre-enrolled observations of one clock as the
        # prior two-heading return. The explicit request is injected on arrival.
        private=ROOT/"identity_stage5_comparison/private"
        self.node=None
        for step in (138,139,140):
            img=cv2.cvtColor(cv2.imread(str(private/f"JEPA_{step}.png")),cv2.COLOR_BGR2RGB)
            box,=json.loads((private/f"JEPA_{step}_boxes.json").read_text())
            roi=tuple(np.array(box,dtype=float)/320)
            native=(np.asarray(roi)*np.tile(img.shape[1::-1],2)).tolist()
            fid=step+1
            center=((np.asarray(roi[:2])+roi[2:])*500).tolist()
            record=a.instance_store.new_observation((*map(float,point),0.),"clock",{"pixel_norm":center,"bbox":native},
                    frame_id=fid,step=0,pixel=tuple(center),bbox=tuple(native))
            if self.node is None:
                self.node=a.instance_store.create_instance(record)
            else:
                a.instance_store.attach_observation(self.node,record)
            ref=SimpleNamespace(roi=roi)
            a._views[record.oid]=(fid,ref)
            a._reference_evidence[record.oid]=ReferenceEvidence("supported",1.,1.)
            a._trusted_pixels[(fid,roi)]=(img,native)
        self.grid,self.point,self.view=grid,point,view
        self.control,self.pitch=control,turn
        self.detector=detector
        self.events=[]
        self.commands=[]
        self.geometry_estimate=None
        self.audits=[]
        self.sent=False
        self.blocked=[]
        self.previous=None
        self.detection_count=None
        self.angles=[]
        self.relooking=False
        a._collect_views=lambda: None
        a._category=lambda obs: "clock"
        a._boxes=self.detect
        a._instance_distance_m=lambda node: math.dist(self.geometry_estimate[:2],self.point)
        a._capture_report_view=lambda: None
        a._note=lambda event, iid=None, **kw:self.events.append(event)
        a.target_instance_id=self.node.iid

    def detect(self,key,rgb,category):
        rows=self.detector.detect(rgb,category)
        self.detection_count=len(rows)
        return rows

    def act(self,obs):
        a=self.agent
        a.now,a._last_observation=int(obs.step_count),obs
        if self.sent:
            if not self.control:
                a._settle_report(obs)
            return int(Action.FINISH)
        if len(self.commands)>=60:
            return int(Action.FINISH)
        x,y,z,w=obs.extras["agent_rotation_xyzw"]
        p=(*floor_xy(obs.gps),yaw(np.quaternion(w,x,y,z)))
        self.geometry_estimate=p
        if self.previous and obs.previous_action==1 and math.dist(p[:2],self.previous[:2])<.05:
            self.blocked.append(self.previous)
        self.previous=p
        if self.control:
            self.sent=True
            return int(Action.TARGET_FOUND)
        path=() if self.relooking else plan_approach(self.grid,p,self.point,1.,.65,self.blocked,observation_view=self.view)
        if path is None:
            return int(Action.FINISH)
        if path:
            primitive=int(path[0])
        elif self.pitch:
            sign=1 if self.pitch>0 else -1
            self.pitch-=sign
            primitive=int(Action.LOOK_UP if sign==1 else Action.LOOK_DOWN)
        else:
            result=SimpleNamespace(target_id=str(self.node.iid))
            # The prior missing-geometry abstention isolates the Stage 7 gate.
            with a._report_intent(obs,result,"arrival"), patch.object(ApproachAgent,"_report_gate",
                    return_value=(False,"identity_fewer_than_two_shared_timestamped_anchors")):
                ok,reason=a._report_gate(self.node,obs)
            self.audits.append({"reason":reason,"detections":self.detection_count,"accepted":bool(ok)})
            if ok:
                self.sent=True
                return a._stage_report(self.node,obs,reason)
            if not self.relooking:
                self.relooking=True
                self.angles=[int(Action.TURN_LEFT),int(Action.TURN_RIGHT),int(Action.TURN_RIGHT)]
            if not self.angles:
                return int(Action.FINISH)
            primitive=self.angles.pop(0)
        self.commands.append(primitive)
        return primitive


def main():
    deny_ip_sockets()
    source,_,manifest,episode=audit(STUDY)
    env,item,data=scene(json.loads(source.read_bytes()))
    det=ProposalWorker(ROOT/"b_verification/env/bin/python",ROOT,"cuda")
    try:
        point,starts,(pitch,initial_pitch),method=observations_for_fixture(env,item,data,episode)
        ref=starts[138]
        grid=SurveyGrid(env.sim.pathfinder,floor_xy(point),ref[1][1])
        threshold=manifest["arguments"]["threshold"]
        starts=outside_starts(env,item,threshold,ref[1],grid)
        assert set(starts)=={"open_approach","obstructed_approach"}
        results=[]
        for label,(pos,delta) in starts.items():
            for control in (True,False):
                target=deepcopy(item)
                target.start_position=pos.tolist()
                target.start_rotation=[0.,0.,0.,1.]
                data.episodes=[target]
                agent=ReturnWithIntent(grid,floor_xy(point),(*floor_xy(ref[1]),ref[2]),det,control,
                                       round((pitch-initial_pitch)/15))
                trial,=evaluate(env,data,agent,max_steps=300,threshold=threshold,
                    action_protocol=episode["action_protocol"],max_steps_per_target=episode["max_steps_per_target"])["episodes"]
                assert not trial["timed_out"]
                results.append(dict(condition=label,immediate_control=control,motion_actions=len(agent.commands),
                    forward_steps=agent.commands.count(1),blocked_forwards=len(agent.blocked),
                    extra_geodesic_distance_m=float(delta),inspection=agent.audits,
                    credited_distinct=trial["found_count"],out_of_range=sum(not e["near_targets"] for e in trial["report_events"]),
                    reports=len(trial["report_events"]),repeats=trial["repeat_report_count"],
                    instance_receipt=bool(agent.agent._confirmed_reference)))
        print(json.dumps(results),flush=True)
        output=dict(scope="Two independent resets of one household clock. Physical actions and RGB, frozen detector, actual Stage 7 arrival, identity and receipt logic, official scorer. No new VLM or full episode.",
            supplied="Same prior three trusted photographs as Stage 6; native intent at estimated arrival, map, precise odometry, depth-based remembered point, and missing-geometry verdict supplied. Only one pre-enrolled instance. Ground truth used for initial range stratification and post-hoc scoring.",
            estimated_target_source=method,paid_calls=0,quota=2,rows=results)
        Path(__file__).with_name("arrival_return_results.json").write_text(json.dumps(output,indent=2)+"\n")
    finally:
        det.close()
        env.close()


if __name__=="__main__":
    main()
