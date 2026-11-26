"""Audit past-view proposals at original decision times without counterfactual navigation."""
from collections import Counter
import json
from pathlib import Path
from source_quality import best,overlap


def read(p):
    return json.loads(p.read_bytes())


def evaluate(project,index,name,spec,with_attributes=False):
    trial=project/"mapping_runtime/pilots/jepa_reid_v24_first2"
    eid=read(trial/"panel.json")["rows"][index]["episode_id"]
    native=Path(read(trial/"comparison/accepted/second"/(eid+".json"))["result"]).parent/"agent"
    events=[json.loads(x) for x in (native/"decision_trace.jsonl").read_text().splitlines()]
    replay=read(Path(__file__).parent/"private_replay"/(name+".json"))
    frames={r["fid"]:r for r in replay["frames"]}
    def measure(fid):
        if fid not in frames:
            return None
        r=frames[fid]; q,ix=best(spec,r["boxes"],518,518)
        hit=ix is not None and any(overlap(r["boxes"][ix]["xyxy"],b)>=.5 for b in r["truth"])
        return dict(visible=bool(r["truth"]),hit=bool(hit),score=q)
    fmetrics={f:measure(f) for f in frames}
    screened={}
    if with_attributes:
        attr=read(Path(__file__).parent/"private_replay"/(name+"_attributes.json"))
        screened={r["fid"]:r["attribute_pass"] for r in attr}
        assert set(screened)==set(frames)
    allstats=Counter(frames=len(frames),visible=sum(r["visible"] for r in fmetrics.values()),
                     high_score=sum(r["score"]>=spec["threshold"] for r in fmetrics.values()),
                     high_score_localized=sum(r["score"]>=spec["threshold"] and r["hit"] for r in fmetrics.values()))
    proposed=Counter(); assisted=Counter()
    inspected=set(); calls=Counter(); done=set()
    proposed_ids=set(); assisted_ids=set()
    for ev in events:
        nav=ev["world_state"]["navigation"]
        current=int(nav.get("current_frame_id") or 0); available=int(nav.get("viewable_frame_id") or 0)
        pool=sorted(fid for fid,r in frames.items() if r["step"]<=ev["step"] and fid<=min(current,available))
        # Reconstructed eligibility is a proxy; the live agent also requires an actual current pose and matching RGB.
        import numpy as np
        unseen=[fid for fid in pool if fid not in inspected]
        for j in np.linspace(0,len(unseen)-1,min(4,len(unseen))).round().astype(int):
            inspected.add(unseen[j])
        usable=sorted((fid for fid in inspected if fid in pool and fid not in done and calls[fid]<2 and screened.get(fid,True) and
                      fmetrics[fid]["score"]>=spec["threshold"]),key=lambda f:(calls[f],-fmetrics[f]["score"],-f))
        offered=usable[:2]
        for fid in offered:
            calls[fid]+=1
        choices=[t["request"].get("frame_id") for t in ev.get("tools",[]) if t["request"].get("name")=="propose_candidates"]
        # Exactly one source-slot per original propose call. Abstention counts as a miss.
        for j,attempt in enumerate(choices):
            old=int(attempt)
            proposed["slots"]+=1; assisted["slots"]+=1; proposed_ids.add(old)
            if old in frames:
                proposed.update(scored=1,visible=int(fmetrics[old]["visible"]),localized=int(fmetrics[old]["hit"]))
            if j<len(offered):
                new=offered[j]; assisted_ids.add(new)
                assisted.update(scored=1,visible=int(fmetrics[new]["visible"]),localized=int(fmetrics[new]["hit"]))
            else:
                assisted["abstained"]+=1
        done.update(int(f) for f in choices if f is not None)
    return dict(category=name,all_mapped_keyframes=dict(allstats),original_source_slots=dict(proposed),
                suggested_source_slots=dict(assisted),distinct_original_sources=len(proposed_ids),
                distinct_assisted_sources=len(assisted_ids),locally_inspected=len(inspected))


if __name__=="__main__":
    dest=Path(__file__).resolve().parent; project=dest.parents[1]
    spec=read(dest/"selector.json")
    out=dict(paid_requests=0,eligible_target=">=103 semantic pixels; tight target bounding box IoU >=0.5",
             assumption="Recorded fixed trajectories. Decision-time mapped and viewable IDs bound a causal pool. Exact per-pose availability is not archived. Original frame picks reuse the same localized-box evaluator.",
             tasks=[evaluate(project,i,n,spec) for i,n in enumerate(("clock","painting"))],
             with_full_goal_filter=[evaluate(project,i,n,spec,True) for i,n in enumerate(("clock","painting"))])
    (dest/"closed_tasks.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out))
