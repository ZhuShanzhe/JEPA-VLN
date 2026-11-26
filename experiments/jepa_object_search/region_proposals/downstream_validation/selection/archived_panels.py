"""Read archived upper-layer proposals and build a private view for RGB spot checking."""
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image


def main(project, output):
    root=project/"mapping_runtime/pilots/jepa_reid_v24_first2"
    doc=json.loads((root/"panel.json").read_bytes())
    eid=doc["rows"][0]["episode_id"]
    receipt=json.loads((root/"comparison/accepted/second"/(eid+".json")).read_bytes())
    path=Path(receipt["result"])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt["result_sha256"]
    rows=[json.loads(line) for line in (path.parent/"agent/candidate_evidence.jsonl").read_bytes().splitlines() if line]
    picks=[r for r in rows if r["event"]=="proposal"]
    assert len({r["candidate_id"] for r in picks})==len(picks)==6
    history=json.loads(path.read_bytes())["episodes"][0]["paper"]["agent_trace"]
    events=[json.loads(line) for line in (path.parent/"agent/confirm_events.jsonl").read_bytes().splitlines() if line]
    end=next(e for e in events if e["event"]=="confirm_view_recheck" and e["reason"]=="no_current_instance_match")
    start=next(r for r in reversed(events[:events.index(end)]) if r["event"]=="confirm_view_start" and r["frame"]==end["frame"] and r["instance"]==end["instance"])
    node=next(n for n in history["final_memory"]["instances"] if n["instance_id"]==end["instance"])
    src,=[r for r in history["observations"] if r["observation_id"] in node["observation_ids"] and r["frame_id"]==end["frame"] and r["step"]<=start["step"]]
    picked={r["frame_id"]:set() for r in picks}
    for entry in rows:
        if entry["event"]=="pick_requested":
            picked.setdefault(int(entry["frame_id"]),set()).update(str(i) for i in entry["mask_ids"])
    output.mkdir(parents=True,exist_ok=True)
    for i,r in enumerate(picks):
        p=path.parent/"agent/candidate_evidence_assets"/r["panel"]
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r["panel"]
        with Image.open(p) as image:
            image.convert("RGB").save(output/f"local_{i+1}.png")
        print(i+1,"registered_reference",r["candidate_id"]==src["candidate_id"],
              "shares_an_explicitly_picked_menu_id",str(r["mask_id"]) in picked[int(r["frame_id"])],
              "status",[e.get("status") for e in rows if e["event"]=="review_receipt" and e.get("candidate_id")==r["candidate_id"]])
    print("Verified six archived upper-layer candidate panels")


if __name__=="__main__":
    main(*map(Path,sys.argv[1:]))
