"""Check a one-step scale delay on the frozen HSSD pair fixtures, with real native identities."""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from check_hssd_review import no_internet, preselect
from delayed_pose import DelayedPoseMixin
from identity import guarded_instance_memory
from trace_agent import TraceFrontier


class CheckAgent(DelayedPoseMixin, TraceFrontier):
    pass


def run(project):
    sys.addaudithook(no_internet)
    kit = project/"jepa_reid_v21"
    weights = kit/"hard_cases/pose_jepa_near_review.pt"
    os.environ.update(NAV_REID_POSE_WEIGHTS=str(weights), NAV_REID_DEVICE="cpu", NAV_COVISIBLE_GUARD="1")
    os.environ.pop("NAV_REID_WEIGHTS", None)
    preds = {r["frame_id"]:r["boxes"] for r in map(json.loads,(kit/"detector/predictions.jsonl").read_text().splitlines())}
    a = CheckAgent.__new__(CheckAgent)
    outcomes = []
    try:
        for kind,(first,i,later,j,*_) in preselect(kit/"data",preds).items():
            a._init_report_memory()
            a.now = 2
            a._metric_snapshot = {"scale":None}
            a._instance_goal_index = {}
            a.instance_dup_radius_m = 0.6
            a.instance_store = guarded_instance_memory()
            a.dup_review_deliver_max = 6
            events = []
            a._audit = lambda event, **fields:events.append(dict(event=event,**fields))
            a._note = lambda *_args, **_kwargs:None
            snaps = [dict(first["snap"],frame_id=1),dict(later["snap"],frame_id=2)]
            a.client = SimpleNamespace(get_frame_points=lambda stride:snaps,
                                       get_frame_image=lambda fid:({"found":True},(first if fid==1 else later)["raw"]))
            a._tool_view_instance = lambda *args,**kwargs:first["raw"]
            obs = [a.instance_store.new_observation(np.asarray(v["points"][k])[[0,2,1]], v["meta"]["category"],
                   frame_id=fid, step=fid, bbox=v["boxes"][k]) for v,k,fid in ((first,i,1),(later,j,2))]
            node = a.instance_store.create_instance(obs[0])
            row = dict(observation_id=obs[1].oid,frame_id=2,first_step=2,
                       neighbors=[dict(instance_id=node.iid,reported=False,dist_m=0.0)])
            a._dup_ledger = {obs[1].oid:row}
            a._last_dup_reviews = [row]
            actual_rows, images, _ = a._dup_review_payload()
            assert actual_rows and not actual_rows[0].get("pose_jepa_review")
            # Substitute only the outer commit and world-render envelopes. The
            # actual duplicate-review payload, feature encoder, scorer and store run.
            with patch.object(TraceFrontier, "_tool_commit_candidates", return_value=dict(
                    duplicate_review=actual_rows,_tool_images=images)):
                initial = a._tool_commit_candidates([])
            assert initial["deferred_until_map_scale"] == [obs[1].oid] and not initial["duplicate_review"]
            reply = a._tool_resolve_duplicate(obs[1].oid,"DUPLICATE",node.iid)
            assert reply["status"] == "waiting_for_map_scale"
            a.now += 1
            a._metric_snapshot = {"scale":1.}
            with patch.object(TraceFrontier, "_build_decider_input", side_effect=lambda *args:
                    (dict(duplicate_review_pending=[dict(row)]), [])):
                state, _ = a._build_decider_input(None)
                repeated, _ = a._build_decider_input(None)
            ready, = state["duplicate_review_pending"]
            scored, = [e for e in events if e["event"]=="pose_review" and e.get("scored")]
            assert state == repeated and a.instance_store.instance_for_observation(obs[1].oid) is None
            assert len(a.instance_store.nodes) == 1 and not node.reported
            hint = "pose_jepa_review" in ready
            assert hint == (kind=="same") and hint == (scored["score"]>=scored["threshold"])
            outcomes.append(dict(label=kind,category=first["meta"]["category"],score=scored["score"],
                                 threshold=scored["threshold"],early_merge_held=True,hint_after_scale=hint,
                                 unique_scoring_calls=1,still_pending=True,reported_instances=0))
    finally:
        if hasattr(a,"_pose_worker"):
            a._pose_worker.pipe.close()
    assert len(outcomes)==2
    return dict(passed=True, paid_requests=0, split="test", scale_delay_steps=1,
                weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(), examples=outcomes)


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1])), indent=2))
