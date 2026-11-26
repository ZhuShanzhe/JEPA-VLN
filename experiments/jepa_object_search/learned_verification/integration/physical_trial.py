"""Free matched Habitat motion from identical RGB evidence and localization preinputs."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import verify_views as scene
import active_agent
from active_view_score import ActiveViewScorer
from weak_view_score import WeakViewScorer
from proposal_worker import ProposalWorker
from run_agent import LearnedVerificationFrontier


P=Path(os.environ["NAV_ROOT"])
def summary(r):
    audit=r["after_action_audit"]
    motion=r["controller_motion"]
    start=next((e for e in r["events"] if e["event"]=="confirm_view_start"),{})
    recheck=[e for e in r["events"] if e["event"]=="confirm_view_recheck"]
    return dict(actions=len(motion),forward=sum(x["action"]==1 for x in motion),
                collided=sum(bool(x["collided"]) for x in motion),plan=start.get("view_source"),
                options=start.get("options"),verification_reason=recheck[-1]["reason"] if recheck else None,
                true_target_pixels_after_return=(audit.get("pixels_each_view") or [0])[-1],
                reference_on_target=audit["reference_task_target"],reported=r["dispatched"],
                official_credit=r["native_viewpoint_audit"]["credited_new_instances"])


def study():
    doc,ep,_,_=scene.case()
    pilot=P/"mapping_runtime/pilots/jepa_reid_v23_first2"
    ticket=json.loads((pilot/"comparison/accepted/second"/(ep["episode_id"]+".json")).read_bytes())
    rows=(Path(ticket["result"]).parent/"agent/decision_trace.jsonl").read_bytes().splitlines()
    boxes=next(json.loads(r)["world_state"]["local_rgb_proposals"]["boxes"] for r in rows if json.loads(r)["step"]==74)
    env,item,data=scene.semantic_scene(doc)
    detector=ProposalWorker(P/"b_verification/env/bin/python",P,"cuda")
    official=active_agent.ActiveFrontier
    result={}
    try:
        frame=scene.replay_until(env,item,data,ep["paper"]["steps"],74)
        ok,jpg=cv2.imencode(".jpg",cv2.cvtColor(scene.rgb(frame),cv2.COLOR_RGB2BGR))
        assert ok
        for arm in ("registered_camera","direct","jepa","context_direct","context_jepa"):
            class Policy(LearnedVerificationFrontier):
                def _begin_confirm_view(self,node,obs,pose,scale):
                    # The prior physical fixture injects the camera pose and registered
                    # crop. Restore the native frame-image lookup to that same real RGB.
                    if not hasattr(self,"_probe_context"):
                        self._probe_context={}
                        self._blocked_probe_edges=[]
                        self._probe_retry=None
                        self._weak_candidate=None
                        self._weak_waiting=[]
                        self._weak_handled=set()
                        context=arm.startswith("context_")
                        scorer=WeakViewScorer if context else ActiveViewScorer
                        family="active_weak" if context else "active"
                        self._verification_scorer=scorer(P/"jepa_reid_v25"/family/"weights"/(arm.split("_")[-1]+".npz"))
                        self.client.get_frame_image=lambda fid: ({"found":fid==74},jpg.tobytes() if fid==74 else b"")
                    return super()._begin_confirm_view(node,obs,pose,scale)
            arm_result={}
            with patch.object(active_agent,"ActiveFrontier",official if arm=="registered_camera" else Policy):
                for name,i,empty in (("true_reference",1,False),("wrong_reference",0,False),("no_live_detection",1,True)):
                    arm_result[name]=summary(scene.execute(env,item,data,ep,boxes[i],i,detector,empty_detection=empty))
            result[arm]=arm_result
        assert all(not x["no_live_detection"]["reported"] for x in result.values())
        assert all(x["true_reference"]["actions"]<=12 for x in result.values())
        assert result["jepa"]["true_reference"]["plan"].startswith("learned_")
    finally:
        detector.close()
        env.close()
    out=P/"jepa_reid_v25/integration/physical_results.json"
    out.write_text(json.dumps(dict(arms=result,paid_calls=0,
        task="One HM3D clock scene, two source boxes, matched synthetic local geometry and ideal odometry"),
        indent=2,allow_nan=False)+"\n")
    print(json.dumps(result))

if __name__=="__main__":
    study()
