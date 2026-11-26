"""Read-only private replay. Standard output contains aggregate acceptance checks."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import Mock

from decision.agent_loop import DecisionLoop
from review_feedback_agent import ReviewFeedbackFrontier
from test_chain import ReportChainTests


def replay():
    p=Path(os.environ["NAV_ROOT"])/"mapping_runtime/pilots/jepa_reid_v24_first2"
    e=json.loads((p/"panel.json").read_bytes())["rows"][1]["episode_id"]
    ticket=json.loads((p/"comparison/accepted/second"/(e+".json")).read_bytes())
    report=Path(ticket["result"])
    assert hashlib.sha256(report.read_bytes()).hexdigest()==ticket["result_sha256"]
    directory=report.parent/"agent"
    load=lambda file:[json.loads(x) for x in (directory/file).read_bytes().splitlines() if x]
    proposals=[r for r in load("candidate_evidence.jsonl") if r["event"]=="proposal"]
    requests=[(r["step"],c) for r in load("decision_trace.jsonl") for c in r.get("tools",[]) if c["request"]["name"]=="commit_candidates"]
    assert len(requests)==2 and requests[0][0]==requests[1][0]
    first,second=[c for _,c in requests]
    assert first["response"]["result"]["rejected"]==["c1","c2"]
    assert second["response"]["result"]["rejected"]==[] and second["response"]["state_changed"] is True
    assert [str(x["candidate_id"]) for x in second["request"]["reviews"]]==["1","2"]
    assert not set(str(x["candidate_id"]) for x in second["request"]["reviews"]) & {r["candidate_id"] for r in proposals}
    f=ReportChainTests()
    f.setUp()
    a=f.make_agent(ReviewFeedbackFrontier)
    with tempfile.TemporaryDirectory() as tmp:
        a.output_dir=tmp
        a._proposal_limit=64
        a._record_rejected_spot=Mock()
        a._record_crosshair_evidence=Mock()
        a._queued_picks,a._queued_pick_results={},[]
        crops={r["candidate_id"]:(directory/"candidate_evidence_assets"/r["panel"]).read_bytes() for r in proposals}
        assert all(hashlib.sha256(crops[r["candidate_id"]]).hexdigest()==r["panel"] for r in proposals)
        a.client.get_candidate_evidence=lambda cid:({"found":True},crops[cid])
        def put(rows):
            for r in rows:
                a._register_picked_candidates({"candidates":[dict(candidate_id=r["candidate_id"],frame_id=r["frame_id"],
                    pixel_norm=r["pixel"],bbox=r["bbox"],mask_id=r["mask_id"])]},"painting",r["goal_index"])
        loop=DecisionLoop(chat_fn=None,tools={"commit_candidates":a._tool_commit_candidates})
        def answer(req):
            raw,_,ok=loop._run_tool(req)
            parsed=json.loads(raw)
            assert parsed["ok"]==ok
            return parsed
        put([r for r in proposals if r["step"]<=requests[0][0]])
        original=answer(first["request"])
        assert original["result"]["rejected"]==first["response"]["result"]["rejected"]
        snapshot=json.dumps(a._proposal_queue,sort_keys=True)
        bad=answer(second["request"])
        assert not bad["ok"] and not bad["state_changed"] and bad["error"]["pending_count"]==0
        assert bad["error"]["code"]=="NO_REVIEWABLE_CANDIDATE"
        assert {r["candidate_id"] for r in bad["error"]["recent_nonpending"]}=={"c1","c2"}
        assert json.dumps(a._proposal_queue,sort_keys=True)==snapshot
        already=answer(first["request"])
        assert already["ok"] and already["result"]["rejected"]==[] and json.dumps(a._proposal_queue,sort_keys=True)==snapshot
        later=[r for r in proposals if r["step"]>requests[0][0]]
        put(later)
        before=json.dumps(a._proposal_queue,sort_keys=True)
        later_feedback=answer(second["request"])["error"]
        assert json.dumps(a._proposal_queue,sort_keys=True)==before
        assert {r["candidate_id"] for r in later_feedback["pending_candidates"]}=={r["candidate_id"] for r in later}
        chosen=later[0]["candidate_id"]
        mixed=answer(dict(name="commit_candidates",reviews=[dict(candidate_id=chosen,verdict="UNCERTAIN"),
                           dict(candidate_id="1",verdict="ACCEPT")]))
        assert mixed["ok"] and mixed["result"]["uncertain"]==[chosen]
        assert mixed["result"]["review_reference_feedback"]["invalid_reviews"]==[dict(candidate_id="1",reason="unknown_candidate_id")]
        assert all(a._proposal_queue[r["candidate_id"]]["status"]=="pending" for r in later[1:])
        assert not a.instance_store.available()
        return dict(paid_calls=0,archived_requests_replayed=len(requests),archived_original_panels_verified=len(crops),
                    exact_real_rejections_preserved=len(original["result"]["rejected"]),unknown_references_explained=len(bad["error"]["invalid_reviews"]),
                    silent_original_success_replaced_by_verified_error=True,pending_at_original_invalid_request=0,
                    later_available_ids_returned_without_mutation=len(later_feedback["pending_candidates"]),
                    mixed_control_valid_row_only=True,known_nonpending_native_behavior_preserved=True,
                    optional_production_class=True,automatic_semantic_decisions=0)


if __name__=="__main__":
    print(json.dumps(replay(),sort_keys=True))
