"""Replay recorded VLM report requests with two handlers on the same physical trajectory."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from agents.instance_store import InstanceMemory
from agents.nav_agent import NavAgent
from benchmark_api import Action
from evaluation.main.evaluator import evaluate
from proposal_worker import ProposalWorker
from replay_clock import Encounter, ROOT, STUDY, audit, rows, scene
from verify_rank import deny_ip_sockets


class Requests(Encounter):
    def __init__(self, path, ep, detector, mode, cls):
        # The stored input/result path implements registration and keyframe images.
        import replay_clock
        with patch.object(replay_clock,"ApproachAgent",cls):
            super().__init__(path,ep,detector)
        self.mode=mode
        self.paid=0
        self.request_rows={t:r for t,r in self.decisions.items() if
                           (r.get("output") or {}).get("action")=="REPORT_FOUND" and
                           r["decision_origin"]=="vlm" and r["validation"]=="ok"}
        self.watermark=-1
        self.extra=0
        self.force_receipt=False
        self.step_map=[]
        self.results=[]
        self.log=[]
        a=self.agent
        a._events=[]
        if mode=="direct":
            a.instance_store=InstanceMemory()
        a._note=lambda name,iid=None,**info:self.log.append(dict(event=name,instance=iid,**info))
        a._stamp_pending_outcome=lambda *args,**kwargs:None
        a._clear_current_target=lambda:None
        a._recover_report=lambda n,obs:(int(Action.TURN_LEFT),"trajectory_held_fixed")
        a._continue_other_target=lambda obs:int(Action.TURN_LEFT)
        a._geometry=lambda n,fid:self.snapshot_geometry(n,fid)
        a._stationary_verdict=self.stationary_result
        self.keyposes={}
        self.official_pos=[]

    def stationary_result(self,fid,ref,obs,boxes):
        row=self.stationary.get((self.at,fid))
        return dict(result="withheld",reason=row["reason"] if row else "missing_two_visible_map_anchors",
                    selected_box=None)

    def snapshot_geometry(self,node,fid):
        p=self.keyposes.get(fid)
        if p is None:
            return None
        return math.dist(p[:2],self.pose[:2])*self.scale, math.dist(p[:2],node.point[:2])*self.scale, \
               math.dist(self.pose[:2],node.point[:2])*self.scale, math.remainder(p[2]-self.pose[2],math.tau)

    def act(self,obs):
        step=int(obs.step_count)
        t=step-self.extra
        self.at=t
        a=self.agent
        self.step_map.append(t)
        if self.force_receipt:
            assert obs.previous_action==int(Action.TARGET_FOUND)
            if self.mode!="direct":
                a._settle_report(obs)
            self.force_receipt=False
        if t>max(self.request_rows):
            return int(Action.FINISH)
        original=self.steps[t]["action"]
        if t==self.watermark:
            return original
        self.watermark=t
        record=self.decisions.get(t)
        if record and record.get("agent_snapshot") and record["world_state"]["navigation"]["current_pose"]:
            self.scale=record["agent_snapshot"]["metric_transform"].get("scale") or self.scale
            self.pose=self.from_navigation(t)
        if self.actions[t]["mapping"]["is_keyframe"]:
            self.capture(t,obs)
            if record and self.pose is not None:
                self.keyposes[self.actions[t]["mapping_frame_id"]]=self.pose
        if record and record.get("agent_snapshot"):
            self.register(t)
        if t not in self.request_rows and (t not in self.poses or self.mode=="direct"):
            return original
        if t in self.poses:
            self.pose=self.poses[t]
        a.now=step
        a._last_observation=obs
        a._last_feed_info=dict(self.actions[t]["mapping"],frame_id=self.actions[t]["mapping_frame_id"])
        if t not in self.request_rows:
            # Recover contemporaneous stationary visual history without inventing a VLM request.
            if (t,13) in self.stationary and not a.instance_store.get(1).reported:
                a._report_gate(a.instance_store.get(1),obs)
            return original
        record=self.request_rows[t]
        nav=record["world_state"]["navigation"]
        active=nav.get("active_target") or {}
        a.target_instance_id=active.get("id") if active.get("type")=="instance" else None
        iid=record["output"]["target_id"]
        node=a.instance_store.get(iid)
        assert isinstance(iid,str) and node is not None
        was_active=a.target_instance_id==node.iid
        tag=len(self.log)
        if self.mode=="direct":
            reply=NavAgent._report_found(a,iid)
        else:
            decision=SimpleNamespace(action="REPORT_FOUND",target_id=iid)
            reply=a._execute_decider_result(obs,decision,record["event"])
        dispatched=reply==int(Action.TARGET_FOUND)
        events=self.log[tag:]
        reasons=[r.get("reason") for r in events if r["event"] in {"report_deferred","report_dispatched"}]
        self.results.append(dict(request_step=t,event=record["event"],active=bool(was_active),
                                 actual_vlm_request=True,reported=dispatched,
                                 reason=reasons[-1] if reasons else None,
                                 request_ordinal=len(self.results)+1))
        if dispatched:
            self.extra+=1
            self.force_receipt=True
        return int(reply) if dispatched else original


def replay_native():
    panel, = json.loads((STUDY/"panel.json").read_bytes())["rows"]
    receipt=json.loads(Path(panel["ref"]).read_bytes())
    path=Path(receipt["result"])
    checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    assert checksum==receipt["result_sha256"]==panel["result_sha256"]
    doc=json.loads(path.read_bytes())
    ep,=doc["episodes"]
    decisions={r["step"]:r for r in rows(path.parent/"agent/decision_trace.jsonl")
               if r.get("decision_origin")=="vlm" and (r.get("output") or {}).get("action")=="REPORT_FOUND"}
    class NativeTape:
        _benchmark_observation_api=True
        def act(self,obs):
            return ep["paper"]["steps"][obs.step_count]["action"]
    env,_,data=scene(doc)
    try:
        replay=evaluate(env,data,NativeTape(),max_steps=ep["max_steps_budget"],
                threshold=doc["manifest"]["arguments"]["threshold"],action_protocol=ep["action_protocol"],
                max_steps_per_target=ep["max_steps_per_target"])
    finally:
        env.close()
    out,=replay["episodes"]
    assert out["steps_taken"]==ep["steps_taken"] and out["found_count"]==ep["found_count"]
    assert [r["action"] for r in out["paper"]["steps"]]==[r["action"] for r in ep["paper"]["steps"]]
    err=max(float(np.linalg.norm(np.array(a["position_before"])-b["position_before"])) for a,b in
            zip(ep["paper"]["steps"],out["paper"]["steps"]))
    assert err<.002
    rr=[]
    for e in out["report_events"]:
        r=decisions[e["step"]]
        assert r["event"]=="arrival" and r["validation"]=="ok"
        rr.append(dict(step=e["step"],explicit_vlm_request=True,
                       active_match=str(r["world_state"]["navigation"]["active_target"]["id"])==r["output"]["target_id"],
                       new_credit=len(e["new_targets"]),incorrect=bool(e["is_false_positive"])))
    summary=dict(mode="native_accepted_action_replay",source_sha256=checksum,requests=rr,
                 steps=out["steps_taken"],physical_prefix_error_m=err,reports=len(rr),
                 credits=out["found_count"],repeats=out["repeat_report_count"],quota=out["required_target_count"],
                 paid_calls=0)
    file=Path(__file__).with_name("clock_native_canonical.json")
    if file.exists():
        assert json.loads(file.read_bytes())==summary
    else:
        file.write_text(json.dumps(summary,indent=2)+"\n")
    print(summary,flush=True)


def main():
    deny_ip_sockets()
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("native","direct","previous","adapted"))
    args=parser.parse_args()
    if args.mode=="native":
        return replay_native()
    path,_,manifest,ep=audit(STUDY)
    if args.mode=="adapted":
        from transfer_agent import ArrivalTransferAgent as Policy
    else:
        from approach_agent import ApproachAgent as Policy
    worker=None if args.mode=="direct" else ProposalWorker(ROOT/"b_verification/env/bin/python",ROOT,"cuda")
    doc=json.loads(path.read_bytes())
    env,_,data=scene(doc)
    try:
        policy=Requests(path,ep,worker,args.mode,Policy)
        scored=evaluate(env,data,policy,max_steps=ep["max_steps_budget"],threshold=manifest["arguments"]["threshold"],
                        action_protocol=ep["action_protocol"],max_steps_per_target=ep["max_steps_per_target"])
    finally:
        env.close()
        if worker: worker.close()
    out,=scored["episodes"]
    tape=out["paper"]["steps"]
    err=max(float(np.linalg.norm(np.asarray(q["position_before"])-ep["paper"]["steps"][i]["position_before"]))
            for q,i in zip(tape,policy.step_map))
    assert err<.002 and not out["timed_out"] and len(tape)==len(policy.step_map)
    reports={policy.step_map[e["step"]]:e for e in out["report_events"]}
    for row in policy.results:
        e=reports.get(row["request_step"])
        assert row["reported"]==bool(e)
        if e:
            row.update(new_credit=len(e["new_targets"]),incorrect=bool(e["is_false_positive"]))
    base=doc["episodes"][0]
    assert base["required_target_count"]==out["required_target_count"]
    official=json.loads((STUDY/"five_comparison.json").read_bytes())["cases"][0]["native"]
    summary=dict(mode=args.mode,physical_prefix_error_m=err,original_trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 paid_calls=0,provided_inputs="actual archived upper-level REPORT_FOUND requests, registrations, current map poses and fixed motion. Live RGB rerendered. Missing map input supplied as withheld. No other decisions regenerated.",
                 native_autonomous={k:official[k] for k in ("found","reports","repeats","steps","quota","success")},
                 realized_stage5_autonomous=dict(reports=len(ep["report_events"]),credited=ep["found_count"]),
                 requests=policy.results,credited_distinct=out["found_count"],reports=len(reports),repeat_reports=out["repeat_report_count"],
                 out_of_range_reports=sum(not e["near_targets"] for e in reports.values()),quota=out["required_target_count"])
    dest=Path(__file__).with_name("clock_"+args.mode+"_canonical.json")
    if dest.exists():
        assert json.loads(dest.read_bytes())==summary
    else:
        dest.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="requests"}),flush=True)


if __name__=="__main__":
    main()
