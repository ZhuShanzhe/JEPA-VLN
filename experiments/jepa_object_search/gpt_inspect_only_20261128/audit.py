"""Audit four paired arms and the separation between GPT inspection and regional routing."""
import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path

GATE_REPORT="33e124299f08ca480efacea50a52a99dc393d732d4b75b554a53c9803d77c36b"
GATE_AUDIT="f86cc533b5255d03defac39e149f4303edd183fd01ec8aadcd1ef92a244f7b19"
MODEL="gpt-4o-2024-11-20"
ARM_LOCK="0bf2ecb49036090a35fac145b4c2f1f8b927e283356583b713b756b770671863"
ARMS=("previous","local","gpt","inspect_only")
METRICS=("distinct_approached","approached_half_m","approached_one_m","already_at_start","quota_met")
PANEL="9d3167d378f2f31fcd7df78d95e222fe150028bcfff739fea0f9334015e69674"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def hashed(p):
    return sha256(p.read_bytes()).hexdigest()


def old_audit(stage):
    module_path=stage.parent/"gpt_region_20261128"/"audit.py"
    assert hashed(module_path)==GATE_AUDIT
    spec=importlib.util.spec_from_file_location("previous_gate_audit",module_path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def totals(rows):
    return {a:{k:sum(r[a][k] for r in rows) for k in METRICS} for a in ARMS}


def field_totals(rows,field):
    counts=Counter()
    for row in rows:
        counts.update(row["diagnostics"][field])
    return dict(counts)


def compare(rows,arm,key):
    differences=[int(r["inspect_only"][key])-int(r[arm][key]) for r in rows]
    win=sum(d>0 for d in differences)
    loss=sum(d<0 for d in differences)
    n=win+loss
    p=min(1.,2*sum(math.comb(n,j) for j in range(min(win,loss)+1))/2**n) if n else 1.
    return dict(improved=win,tied=len(rows)-n,declined=loss,sum_difference=sum(differences),
                mean_difference=sum(differences)/len(rows),exact_two_sided_sign_p=p)


def mechanisms(row,events):
    counts=Counter()
    actions=row["stats"].get("local_steps",0)
    seen_ids=set()
    arrived=set()
    for e in events:
        if e.get("event")=="arrived":
            assert e["goal_id"] in seen_ids and e["goal_id"] not in arrived
            arrived.add(e["goal_id"])
        release=e.get("released") or {}
        if release.get("event")=="arrived":
            assert release["goal_id"] in seen_ids and release["goal_id"] not in arrived
            arrived.add(release["goal_id"])
        if e.get("event")=="acquired":
            assert e["goal_id"] not in seen_ids and e["action_source"]=="local"
            seen_ids.add(e["goal_id"])
        if not e.get("gpt_called"):
            continue
        prior=e["old_selected_region_id"]
        assert prior and prior==e["executed_region_id"],"regional policy was overridden"
        assert e["region_key"]==e["old_selected_region_key"] and e["region_key"]
        shadow=e["gpt_region"]
        assert isinstance(shadow,str) and e["gpt_region_agrees_with_prior"]==(prior==shadow)
        counts["frozen_region_verified"]+=1
        counts["ignored_alternate_region_suggestion"]+=int(prior!=shadow)
        if e["gpt_mode"]=="inspect":
            assert isinstance(e["gpt_patch"],int)
            counts["inspect_requested"]+=1
            counts["inspect_with_camera_site"]+=int(e["exec_status"]=="inspect_runnable")
            counts["inspect_followed_by_local_segment"]+=int(e["action_source"]=="local")
    counts.update(goals_acquired=len(seen_ids),goals_reached=len(arrived),local_physical_actions=actions)
    assert len(seen_ids)==row["stats"].get("acquired",0)
    assert len(arrived)==row["counts"]["releases"].get("arrived",0)
    return dict(counts)


def check_public(report):
    rows=report["by_house"]
    assert len(rows)==8 and len({r["scene"].split("_")[0] for r in rows})==8
    assert report["model"]==MODEL and report["panel_sha256"]==PANEL
    assert report["gate_lock_sha256"]==ARM_LOCK
    assert totals(rows)==report["totals"]
    assert report["pairs"]=={a:{k:compare(rows,a,k) for k in ("distinct_approached","quota_met")} for a in ARMS[:-1]}
    assert report["totals"]["inspect_only"]["distinct_approached"]==sum(report["new_diagnostics"]["first_approach_phase"].values())
    assert sum(report["new_diagnostics"]["physical_actions"].values())==report["actions_per_arm"]==2400
    assert sum(report["new_diagnostics"]["gpt_modes"].values())==report["new_diagnostics"]["mechanisms"]["frozen_region_verified"]
    for field,values in report["new_diagnostics"].items():
        assert values==field_totals(rows,field)


def audit(stage):
    base=stage.parent/"local_target_20261128"/"eval_family"
    gate=stage.parent/"gpt_region_20261128"
    old=old_audit(stage)
    previous=read(gate/"accepted.json")
    assert hashed(gate/"accepted.json")==GATE_REPORT
    old.check_public(previous)
    assert hashed(gate/"gate_lock.json")==previous["gate_lock_sha256"]
    assert hashed(base/"panel.json")==PANEL
    assert hashed(base/"paired_public.json")==old.PREVIOUS_HASH
    panel=read(base/"panel.json")
    paired=read(base/"paired_public.json")["by_house"]
    new_lock=stage/"gate_lock.json"
    assert hashed(new_lock)==ARM_LOCK
    for lock in (base/"local_lock.json",gate/"gate_lock.json",new_lock):
        for rel,h in read(lock).items():
            assert (stage.parent/rel).resolve().is_relative_to(stage.parent) and hashed(stage.parent/rel)==h,rel
    replay=gate/"gpt_gate.py"
    assert read(gate/"gate_lock.json")[str(replay.relative_to(stage.parent))]==hashed(replay)
    assert read(new_lock)[str(replay.relative_to(stage.parent))]==hashed(replay)
    allrows=[]
    for i,s in enumerate(panel):
        path=stage/"runs"/f"{i}_inspect.json"
        current=read(path)
        baseline=previous["by_house"][i]
        expected=s["scene"],s["id"],s["goal"]
        assert tuple(current[k] for k in ("scene","episode","goal"))==tuple(baseline[k] for k in ("scene","episode","goal"))==expected
        shard=stage.parent/"a_hssdov_external"/"eval/hssdov/content"/(s["scene"]+".json.gz")
        assert hashed(shard)==s["shard_sha256"]
        assert current["arm"]=="inspect_only" and current["frozen_sha256"]==hashed(new_lock)
        assert current["steps"]==current["stats"]["steps"]==s["budget"]==300
        assert current["instances_available"]==paired[i]["available"]
        assert current["reachable_instances_at_start"]==paired[i]["reachable"]>=2
        assert current["already_at_start"]==baseline["previous"]["already_at_start"]==0
        assert len(current["first_steps"])==current["distinct_approached"]<=current["approached_half_m"]<=current["approached_one_m"]<=current["instances_available"]
        assert current["quota_met"]==(current["distinct_approached"]>=2)
        events=old.lines(stage/"private_events"/f"{i}.jsonl")
        slug=sha256(str(i).encode()).hexdigest()[:16]
        receipts=old.verify_gate_receipts(stage/"private_events"/f"gpt_{slug}.jsonl",events)
        diagnostics=old.reconcile(current,events)
        diagnostics["gpt_evidence"]=current["upper_decisions"]["gpt_evidence"]
        diagnostics["mechanisms"]=mechanisms(current,events)
        diagnostics["release_reasons"]=current["counts"]["releases"]
        assert diagnostics["gpt_modes"]==current["upper_decisions"]["gpt_mode"]
        row=dict(scene=s["scene"],episode=s["id"],goal=s["goal"],diagnostics=diagnostics,
                 receipts=receipts,verified_result_sha256=hashed(path))
        for a in ARMS[:-1]:row[a]=baseline[a]
        row["inspect_only"]={k:current[k] for k in METRICS+("first_steps",)}
        if not diagnostics["physical_actions"]["local"]:
            row["no_local_steps_matches_region"]={k:current[k]==row["previous"][k] for k in METRICS+("first_steps",)}
        allrows.append(row)
    aggfields=("gpt_modes","gpt_evidence","evidence_by_mode","region_agreement","execution_status","release_reasons",
               "physical_actions","first_approach_phase","mechanisms")
    diagnostics={k:field_totals(allrows,k) for k in aggfields}
    receipt_events=dict(sum((Counter(r["receipts"]["events"]) for r in allrows),Counter()))
    usage=dict(sum((Counter(r["receipts"]["usage"]) for r in allrows),Counter()))
    fees_reported=sum(r["receipts"]["known_cost_entries"] for r in allrows)
    report=dict(model=MODEL,panel_sha256=PANEL,gate_lock_sha256=hashed(new_lock),
                previous_gate_report_sha256=GATE_REPORT,actions_per_arm=2400,totals=totals(allrows),
                pairs={a:{k:compare(allrows,a,k) for k in ("distinct_approached","quota_met")} for a in ARMS[:-1]},
                provider_receipt_events=receipt_events,provider_usage=usage,known_fee_entries=fees_reported,
                known_fee_subtotal=(sum(r["receipts"]["reported_cost_subtotal"] or 0 for r in allrows)
                                    if fees_reported else None),new_diagnostics=diagnostics,by_house=allrows)
    check_public(report)
    old.write_atomic(stage/"accepted.json",report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--stage",type=Path,default=Path(__file__).resolve().parent)
    p.add_argument("--public-only",action="store_true")
    args=p.parse_args()
    if args.public_only:
        check_public(read(args.stage/"accepted.json"))
        print("Verified public paired metrics, stage totals and region controls.")
    else:
        print(json.dumps(audit(args.stage.resolve())["totals"]))
