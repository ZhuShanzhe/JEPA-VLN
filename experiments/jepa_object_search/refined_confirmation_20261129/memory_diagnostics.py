"""Score frozen memory-state transitions, evidence and old-to-new outcomes independently."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/"reactive_view_20261129"


def classify(mask, box):
    if box is None:
        return "no_box"
    l,t,r,b = box
    y,x = np.nonzero(mask)
    inside = int(mask[t:b,l:r].sum())
    if inside==0:
        return "empty_box"
    if len(x)<369:
        return "below_size"
    width=max(0,min(r,x.max()+1)-max(l,x.min()))
    height=max(0,min(b,y.max()+1)-max(t,y.min()))
    inter=width*height
    union=(r-l)*(b-t)+(x.max()+1-x.min())*(y.max()+1-y.min())-inter
    return "correct" if inside>=len(x)*.5 and inter/union>=.30 else "other_localization"


def main():
    transitions=defaultdict(Counter)
    origins=defaultdict(Counter)
    missed=Counter()
    matrix=defaultdict(Counter)
    failure_types=defaultdict(Counter)
    chair=None
    initial_pending={}
    shifts=[]
    for file in sorted((HERE/"private_paired/full").glob("*/*/decision.json")):
        house,source=file.parent.parent.name,file.parent.name
        r=json.loads((PREVIOUS/"private_roll"/house/(source+".json")).read_text())
        assert r["focus"]
        e=r["events"]
        old_at=r["new_confirmation"]
        if old_at is None:
            prior="clear_unreported" if any(step["px"]>=369 for step in e) else "never_clear_unreported"
        else:
            prior="correct" if e[old_at]["correct"] else "false_report"
        now=json.loads(file.read_text())
        with np.load(HERE/"private_scoring"/house/(source+".npz")) as data:
            masks=np.unpackbits(data["packed"],axis=1).reshape(-1,192,192)
        with np.load(HERE/"private_inputs"/house/(source+".npz")) as data:
            odom=data["odom"]
        trace=now["trace"]
        state=[f["state"] for f in trace]
        assert all(not f["accept"] for f in trace if f["state"]=="pending")
        if state[0]=="pending":
            ix=next((i for i,s in enumerate(state) if s=="reanchored"),None)
            if ix is None:
                memory_end="pending_until_stop"
            else:
                dist=float(np.linalg.norm(odom[ix,:2]))
                assert ix>0 and dist >= .12 and all(s=="pending" for s in state[:ix])
                assert all(s=="reanchored" for s in state[ix:])
                shifts.append(dist)
                memory_end="second_view_registered"
            initial_pending[(house,source)] = True
        else:
            assert all(s=="trusted_initial" for s in state)
            memory_end="trusted_initial"
        stop=now["step"]
        outcome="unreported" if stop is None else classify(masks[stop],now["box"])
        aggregate=outcome if outcome in ("unreported","correct") else "false_report"
        transitions[memory_end][aggregate]+=1
        origins[prior][memory_end]+=1
        matrix[prior][aggregate]+=1
        if aggregate=="false_report":failure_types[prior][outcome]+=1
        if prior=="false_report" and e[old_at]["intersection"]==0:
            assert chair is None
            chair={"initial_state":state[0],"terminal_memory_state":memory_end,
                   "old_empty_box_status_after_refit":outcome}
        if prior=="clear_unreported" and aggregate=="unreported":
            visible=[j for j in range(len(trace)) if e[j]["px"]>=369]
            ready=[j for j in visible if trace[j]["state"]!="pending"]
            valid=[j for j in ready if classify(masks[j],trace[j]["box"])=="correct"]
            reason=("memory_pending_on_every_clear_frame" if not ready else
                    "trusted_memory_without_usable_box_on_clear_frames" if not valid else
                    "usable_box_but_score_below_cutoff")
            missed[reason]+=1
    assert sum(v.total() for v in matrix.values())==94 and matrix["correct"].total()==26 and matrix["clear_unreported"].total()==29
    assert chair is not None
    report=dict(new_outcome_by_old_status=matrix,outcome_by_final_memory=transitions,
                final_memory_by_old_status=origins,remaining_prior_29_abstentions=missed,
                distinct_false_reports=failure_types,old_empty_box_chair=chair,
                pending_initial_count=len(initial_pending),second_view_count=len(shifts),
                minimum_measured_displacement_m=min(shifts,default=None))
    (HERE/"memory_diagnostics.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":
    main()
