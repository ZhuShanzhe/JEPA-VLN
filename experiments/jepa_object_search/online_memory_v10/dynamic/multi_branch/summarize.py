import json
import hashlib
from collections import Counter
from pathlib import Path
import sys
import numpy as np


def main(d):
    protocol=json.loads((d/"assets/protocol.json").read_bytes());rooms=protocol["test_houses"];rows=[]
    for h in rooms:
        r=json.loads((d/"metrics"/f"tri_{h:02}.json").read_bytes())
        if len(r)!=7*6*5*2:raise ValueError("incomplete test room")
        for item in r:item["house"]=h
        rows.extend(r)
    stats={}
    for arm in sorted({r["arm"] for r in rows}):
        samples=[r for r in rows if r["arm"]==arm]
        for group,cohort in [("all",samples),("present",[r for r in samples if r["target_count"]>0])]+[
            (name,[r for r in samples if r["target_count"]==n]) for n,name in [(0,"absent"),(1,"single"),(2,"multiple")]]+[
            (f"house_{hh}",[r for r in samples if r["house"]==hh]) for hh in rooms]+[
            (mode,[r for r in samples if r["mode"]==mode]) for mode in sorted({r["mode"] for r in samples})]:
            n=len(cohort);p=[r for r in cohort if r["target_count"]];z=[r for r in cohort if not r["target_count"]]
            def total(fn,ss=cohort):return int(sum(fn(r) for r in ss))
            stats.setdefault(group,{})[arm]=dict(n=n,physical_first=total(lambda r:r["physical_at_first"]),
                physical_within_budget=total(lambda r:r["physically_found"]),correct_report=total(lambda r:r["correct_report"]),
                present_count=len(p),absent_count=len(z),false_absence_if_exhaustion_means_absent=total(lambda r:r["report"] is None,p),
                false_positive_on_absent=total(lambda r:r["report"] is not None,z),
                present_wrong_class_report=total(lambda r:r["report"] is not None and not r["correct_report"],p),
                mean_actions_including_initial_12=float(np.mean([r["actions"] for r in cohort])),
                mean_actions_after_initial_12=float(np.mean([r["actions"]-12 for r in cohort])),
                two_probes=total(lambda r:len(r["visited"])==2))
    assert all(len(set(r["visited"]))==len(r["visited"])<=2 and r["actions"] in (40,108) for r in rows)
    assert len(rows)==1680
    indices={(r["house"],r["mode"],tuple(r["order"]),r["q"],r["arm"]):r for r in rows}
    paired={}
    for a,b in [("latent_update","direct_update"),("latent_update","latent_visit"),("direct_update","direct_visit")]:
        changes=Counter()
        for idx,r in indices.items():
            if idx[-1]!=a:continue
            ref=indices[(*idx[:-1],b)]
            changes["different_first"]+=int(r["visited"][0]!=ref["visited"][0])
            changes["different_history"]+=int(r["visited"]!=ref["visited"])
            changes["won_correct_report"]+=int(r["correct_report"] and not ref["correct_report"])
            changes["lost_correct_report"]+=int(ref["correct_report"] and not r["correct_report"])
        paired[a+"_vs_"+b]=dict(changes)
    report=dict(task="real three-bay Habitat simulation, both decisions executed in same session",
        train_houses=6,train_configurations=252,test_houses=4,test_configurations=168,test_queries_per_arm=336,
        model_runs=len(rows),attempts_max=2,goal_visible_initial_pixels=0,eligible_mask_pixels_min=64,
        start_panorama_steps=12,outward_steps=28,return_steps=40,second_outward_steps=28,maximum_actions=108,
        weight_sha256=hashlib.sha256((d/"frozen.npz").read_bytes()).hexdigest(),
        stats=stats,paired=paired,
        limits="Two named categories, red-chair distractor; static placement, engineered cues, four held-out room layouts. Exhausted probes warrant not_found rather than certified global absence.")
    (d/"summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(stats["all"],indent=2))


if __name__=="__main__":main(Path(sys.argv[1]))
