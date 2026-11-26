"""Replay real RGB through the selectable agent, its native proposal contract and its visual gate."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from PIL import Image
from agents.nav_agent import NavAgent
from appearance import build_reference,match_reference
from report_evidence import current_support,eligible_boxes
from provider_agent import BoxComparisonAgent
from test_chain import ReportChainTests


def load(path):
    return {r["frame_id"]:r for r in map(json.loads,path.read_bytes().splitlines())}


def verify(project):
    data=project/"jepa_box_v26/habitat/private"
    manifest=load(data/"manifest.jsonl")
    # These three frames were fixed before scoring: an initial clue, a near view,
    # and an archived art frame. No labels are loaded by the providers.
    source="clock_archived_registered"
    near="clock_archived_after_approach"
    art=next(fid for fid,r in manifest.items() if r["category"]=="wall_art" and r["group"]=="archived")
    import sys
    sys.path.insert(0,str(project/"jepa_box_v26/habitat"))
    from clock_audit import registration_box
    registered=registration_box(project)
    initial=np.array(Image.open(data/manifest[source]["rgb"]).convert("RGB"))
    reference=build_reference(initial,registered)
    assert reference is not None
    output={}
    for provider in ("native","dino","refined"):
        with patch.dict(os.environ,NAV_BOX_PROVIDER=provider,NAV_LOCAL_BOXES="1"),TemporaryDirectory() as folder:
            f=ReportChainTests()
            f.setUp()
            agent=f.make_agent(BoxComparisonAgent)
            agent.output_dir=folder
            # No upper request is sent. For the native mode the original proposer
            # must keep its mask IDs; the extension must not resubmit point prompts.
            agent._proposal_queue={}
            reference_rows=load(data/({"native":"sam","dino":"dino","refined":"refined"}[provider]+".jsonl"))
            if provider=="native":
                ranked=load(project/"jepa_box_v26/compare/private/habitat_sam.jsonl")
            result={"frames_checked":0,"source_registration_unchanged":False}
            try:
                for key,fid in enumerate((source,near,art),1):
                    item=manifest[fid]
                    image=np.array(Image.open(data/item["rgb"]).convert("RGB"))
                    category=item["category"].replace("wall_art","painting")
                    if provider=="native":
                        exp=[r for r in ranked[fid]["boxes"][:3] if r["review_margin"]>0]
                    else:
                        exp=reference_rows[fid]["boxes"]
                    loc=agent._boxes(("live",key),image,category)
                    if agent._detector_failed:
                        raise RuntimeError("Agent box worker failed: "+provider)
                    assert len(loc)==len(exp),(provider,fid,len(loc),len(exp))
                    assert agent._boxes(("live",key),image,category) is loc
                    for a,b in zip(loc,exp):
                        np.testing.assert_allclose(a["bbox"],b["bbox"] if "bbox" in b else b["xyxy"],atol=.05,rtol=0)
                        if provider=="native":
                            assert a["mask_id"]==b["mask_id"]
                    if fid==near:
                        boxes=eligible_boxes(loc,image.shape)
                        match,reason=current_support(reference,match_reference(reference,image),image,boxes)
                        result.update(near_live_proposals=len(loc),near_match=match is not None,near_reason=reason)
                    result["frames_checked"]+=1
                if provider=="native":
                    original={"masks":[{"mask_id":3},{"mask_id":9}],"stopped":False}
                    # Exercise the project's actual supplemental-proposal hook
                    # after returning a frozen mapping-service mask table.
                    from multiview_agent import MultiviewFrontier
                    with patch.object(NavAgent,"_tool_propose_candidates",return_value=original):
                        response=MultiviewFrontier._tool_propose_candidates(agent,1,"clock")
                    assert response["masks"]==original["masks"] and "local_pending_proposals" not in response
                    result["original_menu_passthrough"]=True
                assert not agent.instance_store.available() and not agent._pending_report
                result["source_registration_unchanged"]=True
            finally:
                if hasattr(agent,"_proposal_worker"):
                    agent._proposal_worker.close()
            output[provider]=result
    path=project/"jepa_box_v26/adapter/interface_results.json"
    path.write_text(json.dumps({"paid_requests":0,"arms":output},indent=2)+"\n")
    print(output)


if __name__=="__main__":
    verify(Path(os.environ["NAV_ROOT"]))
