"""Count stages using distinct native IDs and actual successful client attachments."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def read_jsonl(path):
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            yield from (json.loads(line) for line in stream if line.strip())


def summarize(agent_dir):
    calls, selected, queued = Counter(), set(), set()
    in_state, sent, submitted, old_fails = set(), set(), set(), set()
    for decision in read_jsonl(agent_dir/"decision_trace.jsonl"):
        for t in decision.get("tools", []):
            req = t["request"]
            name = req.get("name")
            calls[name] += 1
            resp = t.get("response", {}).get("result", {})
            if name == "pick_segment":
                selected.add(req["frame_id"])
                if resp.get("queued"):
                    queued.add(req["frame_id"])
            if name == "commit_candidates":
                submitted.update(str(r["candidate_id"]) for r in req.get("reviews", []) if r.get("candidate_id"))
        for item in decision.get("world_state", {}).get("queued_pick_results", []):
            in_state.update(r["candidate_id"] for r in item.get("proposals", []))
            if "candidate registration is not available yet" in str(item.get("error", "")):
                old_fails.add(item["frame_id"])
    for row in read_jsonl(agent_dir/"vlm_calls.jsonl"):
        if row.get("kind") == "decision" and row.get("ok"):
            sent.update(i["label"][len("proposal_"):] for i in row.get("images", [])
                        if i.get("label","").startswith("proposal_"))
    log = (agent_dir/"events.log").read_text(encoding="utf-8") if (agent_dir/"events.log").exists() else ""
    completed = [(int(fid), int(n)) for fid,n in re.findall(r"queued pick on frame (\d+) completed: (\d+) proposal",log)]
    evidence = list(read_jsonl(agent_dir/"candidate_evidence.jsonl"))
    unique = lambda tag: len({r["candidate_id"] for r in evidence if r["event"] == tag and r.get("candidate_id")})
    events = Counter(r["event"] for r in evidence)
    projected = {r["candidate_id"]:r for r in evidence if r["event"]=="proposal"}
    available_panels = {cid for cid,r in projected.items() if r.get("panel")}
    available_panels.update(r["candidate_id"] for r in evidence if r["event"]=="original_panel_recovered")
    requests = {r["candidate_id"] for r in evidence if r["event"]=="review_requested"}
    outstanding = set(projected)-requests
    unreviewed = {"original_panel_unavailable":len(outstanding-available_panels),
                  "panel_available_without_successful_client_attachment":len((outstanding&available_panels)-sent),
                  "successful_client_attachment_without_review":len(outstanding&sent)}
    return dict(region_propose_calls=calls["propose_candidates"],pick_calls=calls["pick_segment"],
                selected_frames=len(selected),ever_queued_frames=len(queued),
                native_async_completed_batches=len(completed),native_async_proposal_rows=sum(n for _,n in completed),
                native_mapping_not_ready_delivered_frames=len(old_fails),
                queued_state_distinct_candidates=len(in_state),successful_client_distinct_panels=len(sent),
                commit_calls=calls["commit_candidates"],submitted_distinct_candidates=len(submitted),
                v24_distinct_proposals=unique("proposal") if evidence else None,
                v24_distinct_reoffered=unique("pending_review_in_input") if evidence else None,
                v24_unreviewed_at_last_record=unreviewed if evidence else None,
                v24_event_counts=dict(events) if evidence else None,
                v24_distinct_registered=len({(inst["instance_id"]) for r in evidence if r["event"]=="review_receipt" for inst in r["instances"]}) if evidence else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_dir",type=Path)
    print(json.dumps(summarize(parser.parse_args().agent_dir),sort_keys=True))
