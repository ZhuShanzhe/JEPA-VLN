"""Audit the frozen multiview policy, paired approach events, per-camera grounding and two provider ledgers."""
from collections import Counter
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PRIOR=ROOT/"committed_inspect_20261128"
sys.path.insert(0,str(PRIOR))
from audit_committed import (METRICS, MODEL, PANEL, ARMS, compare_prefix, local_events, lines, read,
                             digest, trajectory, validate_receipts, public_checks as previous_checks)

EXPECTED_PRIOR="7d3c92d32245a3fdd756a34b7567581162fc8bf355de95016ac612fed460e618"


def reconcile_cue(directory,decisions,trace):
    checked=[e for e in decisions if e.get("cue_checked")]
    receipts=lines(directory/"cue_decisions.jsonl") if (directory/"cue_decisions.jsonl").exists() else []
    calls=lines(directory/"cue_calls.jsonl") if (directory/"cue_calls.jsonl").exists() else []
    final=receipts[-len(checked):] if checked else []
    assert len(final)==len(checked)
    states={}
    choices={}
    meter=Counter()
    costs=[]
    for row in calls:
        assert row["model"]==MODEL and len(row["request_hash"])==64
        key=row["request_hash"]
        if row["event"]=="dispatch":
            assert key not in states or (states[key]["event"]=="error" and states[key]["retryable_denial"])
        else:
            assert key in states and states[key]["event"]=="dispatch" and row["event"] in ("success","error")
            if row["event"]=="success":
                choices[key]=row["choice"]
            if row.get("total_tokens") is not None:
                meter["total_tokens"]+=int(row["total_tokens"])
            if row.get("cost") is not None: costs.append(float(row["cost"]))
        states[key]=row
        meter[row["event"]]+=1
    prior_choices={r["request_hash"]:r["choice"] for r in
                   [read(p) for p in (PRIOR/"private_dino/gpt_multiview").glob("*.response.json")]}
    funnel=Counter()
    for e,r in zip(checked,final):
        assert e["gpt_mode"]=="inspect" and e["gpt_called"] and r["model"]==MODEL and not e["commit_previous_id"]
        assert e["cue_event"]==r["event"]
        key,choice=r["request_hash"],r["choice"]
        kind=r["event"]
        assert kind in ("no_candidates","success","prior_cache","run_cache")
        assert bool(choice)==e["cue_selected"]
        funnel["invoked"]+=1
        funnel[kind]+=1
        funnel["with_candidates"]+=r["candidates"]>0
        if kind=="no_candidates":
            assert r["candidates"]==0 and choice is None and key is None
        else:
            assert r["candidates"]>0 and len(key)==64
            verified=prior_choices if kind=="prior_cache" else choices
            assert key in verified and choice==verified[key]
        if choice is None:
            assert e["action_source"]!="local" and e["anchors"]==e["sites"]==0 and e["cue_view_index"] is None
            funnel["no_local_dispatch_after_refusal"]+=1
            continue
        idx=e["cue_view_index"]
        assert type(idx) is int and idx in (0,1,2) and "RCL"[idx]==choice[0] and choice==e["cue_candidate_id"]
        left,top,right,bottom=e["cue_bbox_norm"]
        assert 0<=left<right<=1 and 0<=top<bottom<=1
        assert e["cue_rgbd_aligned"] and e["cue_anchor_source"]=="selected_view_aligned_rgbd"
        assert abs(math.remainder(e["physical_scan"][idx]["pose"][2]-e["cue_projection_yaw"],math.tau))<1e-6
        funnel["selected_box"]+=1
        funnel["selected_"+("right","center","left")[idx]]+=1
        funnel["accepted_differs_from_upper_view_hint"]+=idx!=e["gpt_patch"]
        funnel["projected"]+=e["anchors"]>0
        funnel["with_reachable_view"]+=e["sites"]>0
        if e["sites"]:
            assert e["cue_score_source"]=="selected_box_and_ordered_scan_jepa" and e["anchors"]>0
        if e["action_source"]=="local":
            assert e["event"]=="acquired" and e["sites"]>0 and e["cue_anchor_xy"]==e["anchor_xy"]
    scans=0
    for e in decisions:
        if "physical_scan" not in e:
            assert not e.get("gpt_called") and not e.get("cue_checked")
            continue
        scans+=1
        frame=e["physical_scan"]
        end=e["step"]
        assert [x["step"] for x in frame]==[end-3,end-4,end-1]
        assert [trace[k]["action"] for k in range(end-3,end+1)]==["turn_right","turn_left","turn_left","turn_right"]
        for f,angle in zip(frame,(-30,0,30)):
            t=trace[f["step"]]
            x,y,yaw=f["pose"]
            assert max(abs(x-t["xyz"][0]),abs(y+t["xyz"][2]),abs(t["yaw"]-yaw))<1e-5
            assert len(f["rgb_sha256"])==len(f["depth_sha256"])==64
            assert abs(math.remainder(yaw-trace[end]["yaw"]-math.radians(angle),math.tau))<1e-4
        if e.get("gpt_mode")=="inspect": assert e["gpt_patch"] in (0,1,2)
        if e.get("event")=="acquired": assert e["cue_checked"] and e["cue_selected"]
    funnel["verified_physical_triplets"]=scans
    return dict(funnel),dict(meter),costs


def aggregate(rows,field):
    total=Counter()
    for r in rows: total.update(r[field])
    return dict(total)


def paired(rows,arm,metric):
    ds=[int(r["multiview"][metric])-int(r[arm][metric]) for r in rows]
    w,l=sum(d>0 for d in ds),sum(d<0 for d in ds)
    n=w+l
    return dict(wins=w,ties=len(ds)-n,losses=l,delta=sum(ds),
                exact_two_sided_sign_p=min(1.,2*sum(math.comb(n,k) for k in range(min(w,l)+1))/2**n) if n else 1.)


FIELDS=("mechanisms","release_reasons","first_approach_phase","physical_actions","cue_funnel","cue_journal","upper_journal",
        "upper_sources","upper_usage","upper_modes","phase_025m")


def check_public(report):
    rows=report["by_house"]
    assert len(rows)==len({r["scene"].split("_")[0] for r in rows})==8
    assert report["model"]==MODEL and report["panel"]==PANEL and report["previous_report"]==EXPECTED_PRIOR
    for arm in (*ARMS,"multiview"):
        assert report["totals"][arm]=={k:sum(r[arm][k] for r in rows) for k in METRICS}
    for field in FIELDS:
        assert report[field]==aggregate(rows,field)
    assert sum(report["physical_actions"].values())==2400
    assert sum(report["first_approach_phase"].values())==report["totals"]["multiview"]["distinct_approached"]
    assert report["paired"]=={a:{k:paired(rows,a,k) for k in ("distinct_approached","quota_met")} for a in ARMS}
    assert report["mechanisms"]["local_actions"]==report["physical_actions"]["local"]
    assert report["cue_funnel"].get("selected_box",0)>=report["mechanisms"]["acquired"]
    assert report["fee_entries"]==sum(r["fee_entries"] for r in rows)
    assert report["reported_fee_subtotal"]==(sum(r["reported_fee_subtotal"] or 0 for r in rows) if report["fee_entries"] else None)


def audit():
    previous=read(PRIOR/"accepted.json")
    assert digest(PRIOR/"accepted.json")==EXPECTED_PRIOR
    previous_checks(previous)
    gate_path=ROOT/"gpt_region_20261128/audit.py"
    assert digest(gate_path)=="f86cc533b5255d03defac39e149f4303edd183fd01ec8aadcd1ef92a244f7b19"
    loader=importlib.util.spec_from_file_location("frozen_event_audit",gate_path)
    historic=importlib.util.module_from_spec(loader)
    loader.loader.exec_module(historic)
    panel_path=ROOT/"local_target_20261128/eval_family/panel.json"
    assert digest(panel_path)==PANEL
    lock=HERE/"gate_lock.json"
    for rel,h in read(lock).items():
        assert (ROOT/rel).resolve().is_relative_to(ROOT) and digest(ROOT/rel)==h,rel
    rows=[]
    for i,s in enumerate(read(panel_path)):
        oldrow=previous["by_house"][i]
        result=HERE/"runs"/f"{i}_multiview.json"
        new=read(result)
        assert tuple(new[k] for k in ("scene","episode","goal"))==(s["scene"],s["id"],s["goal"])
        assert new["arm"]=="multiview" and new["frozen_sha256"]==digest(lock)
        assert new["steps"]==new["stats"]["steps"]==s["budget"]==300
        assert new["quota_met"]==(new["distinct_approached"]>=2) and new["already_at_start"]==0
        assert len(new["first_steps"])==new["distinct_approached"]<=new["approached_half_m"]<=new["approached_one_m"]<=new["instances_available"]
        control=PRIOR/"runs"/f"{i}_committed.json"
        assert digest(control)==oldrow["verified_result_sha256"]
        old=read(control)
        assert old["instances_available"]==new["instances_available"] and old["reachable_instances_at_start"]==new["reachable_instances_at_start"]>=2
        private=HERE/"private_events"
        logpath,tracepath=private/f"{i}.jsonl",private/f"{i}.poses.jsonl"
        assert digest(logpath)==new["private_events_sha256"] and digest(tracepath)==new["private_trace_sha256"]
        events,trace=lines(logpath),lines(tracepath)
        trajectory(trace)
        diagnostics=historic.reconcile(new,events)
        assert new["upper_decisions"]["gpt_mode"]==diagnostics["gpt_modes"]
        mechanism,releases=local_events(new,events)
        name="gpt_"+sha256(str(i).encode()).hexdigest()[:16]+".jsonl"
        receipts=validate_receipts(historic,PRIOR/"private_events"/name,private/name,events)
        assert sum(receipts["final_response_sources"].values())==sum(diagnostics["gpt_modes"].values())
        funnel,meter,costs=reconcile_cue(private/f"cue_{i}",events,trace)
        nfees=receipts["cost_entries"]+len(costs)
        assert sum(1 for e in events if e.get("event")=="acquired")==mechanism["acquired"]
        data=dict(scene=s["scene"],goal=s["goal"],episode=s["id"],verified_result_sha256=digest(result),
                  mechanisms=mechanism,release_reasons=releases,upper_modes=diagnostics["gpt_modes"],
                  first_approach_phase=diagnostics["first_approach_phase"],physical_actions=diagnostics["physical_actions"],
                  phase_025m={k:v["0.25"] for k,v in new["distance_by_phase"].items()},
                  cue_funnel=funnel,cue_journal=meter,upper_journal=receipts["new_journal_events"],
                  upper_sources=receipts["final_response_sources"],upper_usage=receipts["new_usage"],
                  fee_entries=nfees,reported_fee_subtotal=((receipts["reported_cost_subtotal"] or 0)+sum(costs)) if nfees else None,
                  upper_prefix_comparison=compare_prefix(events,lines(PRIOR/"private_events"/f"{i}.jsonl")))
        for a in ARMS: data[a]=oldrow[a]
        data["multiview"]={k:new[k] for k in METRICS+("first_steps",)}
        rows.append(data)
    nfees=sum(r["fee_entries"] for r in rows)
    report=dict(model=MODEL,panel=PANEL,gate_lock=digest(lock),previous_report=EXPECTED_PRIOR,budget_per_house=300,
                task="Approach distinct instances' reference viewpoints; correct reporting not evaluated",
                totals={a:{k:sum(r[a][k] for r in rows) for k in METRICS} for a in (*ARMS,"multiview")},
                paired={a:{k:paired(rows,a,k) for k in ("distinct_approached","quota_met")} for a in ARMS},
                fee_entries=nfees,reported_fee_subtotal=sum(r["reported_fee_subtotal"] or 0 for r in rows) if nfees else None,
                by_house=rows,**{field:aggregate(rows,field) for field in FIELDS})
    check_public(report)
    historic.write_atomic(HERE/"accepted.json",report)
    return report


if __name__=="__main__":
    if sys.argv[1:]==["--public-only"]:
        check_public(read(HERE/"accepted.json"))
        print("summary verified")
    else:
        print(json.dumps(audit()["totals"]))
