"""Physical HM3D dispatch under scripted trigger and cross-camera cue selection."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from multi_agent import MultiViewAgent
import multi_agent
import multi_policy

HERE=Path(__file__).resolve().parent


class Intent:
    calls=0
    def __init__(self,ep,where):
        self.path=Path(where)/"fixture.json"

    def choose(self, **kw):
        self.calls+=1
        mode="explore" if self.calls==1 else "inspect"
        return dict(mode=mode,patch_id=1 if mode=="inspect" else None,region_id=kw["regions"][-1]["id"],
                    target_evidence="suspected",cue="occlusion")


class LeftCue:
    def __init__(self,where):
        self.last_receipt={"event":"controlled"}
    def select(self,views,goal):
        assert len(views)==3
        return dict(view_index=2,bbox_norm=[.4,.40,.6,.60],candidate_id="L1",category=goal)
    def close(self):
        pass


def main():
    spec=importlib.util.spec_from_file_location("native_host",HERE.parent/"local_target_20261128/check_host.py")
    host=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(host)
    facts=[]
    original=MultiViewAgent._bind_choice
    def capture(agent,obs,scan,native,decision,region):
        if decision.get("cue_selected"):
            assert decision["gpt_patch"]==1 and decision["cue_view_index"]==2
            assert decision["cue_anchor_source"]=="source_view_observed_map_column_proxy"
            assert not decision["cue_rgbd_aligned"] and decision["cue_anchor_xy"] is not None
            assert decision["cue_score_source"]=="selected_box_and_ordered_scan_jepa"
            facts.append(decision["action_source"])
        return original(agent,obs,scan,native,decision,region)
    with tempfile.TemporaryDirectory() as td:
        result_file=Path(td)/"receipt.json"
        with patch.dict(os.environ,{"NAV_REQUIRE_DECISION_PREFLIGHT":"0"}), patch.object(
                multi_agent,"MultiGate",Intent), patch.object(multi_agent,"NativeCue",LeftCue), patch.object(
                MultiViewAgent,"_bind_choice",capture):
            host.run(result_file)
        receipt=json.loads(result_file.read_text())
    assert facts==["local"]
    result={k:receipt[k] for k in ("agent_factory","real_rgb","controlled_map_and_pose","paid_requests",
              "acquired_distance_m","retained_distance_m","native_finish_preempted","scale_reset")}
    result.update(scripted_gate_and_box=True,gate_view="center",selected_camera="left",
                  projection="source_view_observed_map_column_proxy",used_metric_depth=False,
                  checked_real_jepa_and_follower=True)
    (HERE/"native_check.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    main()
