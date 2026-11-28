"""Verify all frozen paired episodes before summarizing the GPT high-level policy."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path


PANEL_HASH = "9d3167d378f2f31fcd7df78d95e222fe150028bcfff739fea0f9334015e69674"
PREVIOUS_HASH = "e1b7c205676ad71175f09bc3bede8f3c329cf58cbf1982187e1c07e9dab17edc"
MODEL = "gpt-4o-2024-11-20"
ARMS = ("previous", "local", "gpt")
METRICS = ("distinct_approached", "approached_half_m", "approached_one_m", "already_at_start", "quota_met")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def totals(rows):
    return {arm:{m:sum(r[arm][m] for r in rows) for m in METRICS} for arm in ARMS}


def paired(rows, other, metric):
    diffs=[int(r["gpt"][metric])-int(r[other][metric]) for r in rows]
    wins=sum(d>0 for d in diffs)
    losses=sum(d<0 for d in diffs)
    n=wins+losses
    p=min(1.,2.*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/(2**n)) if n else 1.
    return dict(improved=wins,tied=diffs.count(0),declined=losses,
                mean_paired_difference=sum(diffs)/len(diffs),two_sided_exact_sign_p=p)


def reconcile(row, events):
    stats=row["stats"]
    decisions=[e for e in events if "action_source" in e]
    assert len(decisions)==stats["decisions"]
    modes=Counter()
    region_agreement=Counter()
    statuses=Counter()
    evidence_by_mode=Counter()
    durations=Counter()
    hit_phase=Counter()
    intervals=[]
    for e in decisions:
        a=e["step"]
        b=e.get("last_step",a)
        assert 0<=a<=b<=300
        if intervals:
            assert intervals[-1][1]<=a,"overlapping outcomes or untrimmed retry trace"
        intervals.append((a,b,e["action_source"]))
        durations[e["action_source"]]+=b-a
        if e.get("gpt_called"):
            modes[e["gpt_mode"]]+=1
            agrees=e["gpt_region_agrees_with_prior"]
            assert isinstance(agrees,bool)
            region_agreement["same" if agrees else "different"]+=1
            evidence_by_mode[e["gpt_mode"]+"_"+e["gpt_evidence"]]+=1
        status=e.get("exec_status")
        if status:
            statuses[status]+=1
    for s in row["first_steps"]:
        labels=[phase for a,b,phase in intervals if a<s<=b]
        assert s==0 or len(labels)==1,("unattributed hit",s,labels)
        hit_phase["start" if s==0 else labels[0]]+=1
    budget={p:stats.get(p+"_steps",0) for p in ("region","local","scan","tail","empty")}
    assert sum(budget.values())==300
    for p in ("region","local"):
        assert durations[p]==budget[p]
    return dict(decisions=len(decisions),gpt_modes=dict(modes),region_agreement=dict(region_agreement),
                evidence_by_mode=dict(evidence_by_mode),execution_status=dict(statuses),
                physical_actions=budget,first_approach_phase=dict(hit_phase),
                local_targets_acquired=stats.get("acquired",0),
                local_targets_reached=row.get("counts",{}).get("releases",{}).get("arrived",0))


def aggregate_counters(per_house, field):
    c=Counter()
    for row in per_house:
        c.update(row["gate_diagnostics"].get(field,{}))
    return dict(c)


def verify_gate_receipts(path, events):
    decisions=[e for e in events if e.get("gpt_called")]
    receipts=lines(path) if path.exists() else []
    states={}
    delivered=[]
    spend=Counter()
    samples=Counter()
    known_cost=[]
    for e in receipts:
        k=e["hash"]
        kind=e["event"]
        spend[kind]+=1
        if kind=="dispatch":
            assert e["model"]==MODEL
            assert states.get(k) not in ("dispatch","transport_uncertain")
        elif kind=="cache":
            assert states.get(k) in ("success","cache")
        elif kind in ("success","invalid_decision","gateway_http","transport_uncertain","wrong_model"):
            assert states.get(k)=="dispatch",(kind,states.get(k))
        if kind in ("success","invalid_decision"):
            assert e["model"]==MODEL
            for key in ("input_tokens","output_tokens","total_tokens"):
                if e.get(key) is not None:
                    assert int(e[key])>=0
                    samples[key]+=int(e[key])
                else:
                    samples[key+"_missing"]+=1
            cost=e.get("provider_reported_cost")
            if cost is not None:
                known_cost.append(float(cost))
        if kind in ("success","cache"):
            delivered.append(e["decision"])
        states[k]=kind
    assert len(delivered)>=len(decisions)
    # Logs from interrupted attempts are retained. The last completed attempt must correspond
    # to the final sequence of returned decisions, including replayed cached answers.
    selected=delivered[len(delivered)-len(decisions):] if decisions else []
    for result,ev in zip(selected,decisions):
        assert result["mode"]==ev["gpt_mode"] and result["region_id"]==ev["gpt_region"]
        assert result["patch_id"]==ev["gpt_patch"] and result["target_evidence"]==ev["gpt_evidence"]
        assert result["cue"]==ev["gpt_cue"]
    return dict(events=dict(spend),usage=dict(samples),known_cost_entries=len(known_cost),
                reported_cost_subtotal=sum(known_cost) if known_cost else None,
                returned_successfully_outside_final_attempt=len(delivered)-len(decisions))


def write_atomic(path,obj):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    tmp.replace(path)


def check_public(report):
    rows=report["by_house"]
    assert len(rows)==report["saved_complete_episodes"]==8
    assert len({r["scene"].split("_")[0] for r in rows})==8
    assert report["panel_sha256"]==PANEL_HASH and report["model"]==MODEL
    assert totals(rows)==report["totals"]
    for arm in ("previous","local"):
        for metric in ("quota_met","distinct_approached"):
            assert paired(rows,arm,metric)==report["paired"][arm][metric]
    for field in ("gpt_modes","region_agreement","execution_status","physical_actions","first_approach_phase"):
        assert aggregate_counters(rows,field)==report["gpt_diagnostics"][field]
    assert sum(report["gpt_diagnostics"]["physical_actions"].values())==report["action_budget_per_arm"]==2400
    assert sum(report["gpt_diagnostics"]["first_approach_phase"].values())==report["totals"]["gpt"]["distinct_approached"]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--status",action="store_true")
    parser.add_argument("--public-only",action="store_true")
    args=parser.parse_args()
    stage=args.stage.resolve()
    if args.public_only:
        check_public(read(stage/"accepted.json"))
        print("Verified public paired metrics, decision and action totals.")
        return
    base=stage.parent/"local_target_20261128"/"eval_family"
    assert digest(base/"panel.json")==PANEL_HASH
    assert digest(base/"paired_public.json")==PREVIOUS_HASH
    panel=read(base/"panel.json")
    assert len({s["scene"].split("_")[0] for s in panel})==len(panel)==8
    outpaths=[stage/"runs"/f"{i}_gpt.json" for i in range(len(panel))]
    missing=[i for i,p in enumerate(outpaths) if not p.exists()]
    if args.status:
        print(json.dumps(dict(saved=len(outpaths)-len(missing),outstanding=missing)))
        return
    if missing:
        raise RuntimeError("Waiting for episodes "+str(missing))
    seal=stage/"gate_lock.json"
    for rel, expected in read(seal).items():
        assert digest(stage.parent/rel)==expected,rel
    prior=read(base/"paired_public.json")
    summaries=[]
    for i,(spec,stored,op) in enumerate(zip(panel,prior["by_house"],outpaths)):
        row=read(op)
        assert row["arm"]=="gpt_region" and row["frozen_sha256"]==digest(seal)
        assert (row["scene"],row["episode"],row["goal"])==(spec["scene"],spec["id"],spec["goal"])
        assert row["steps"]==row["stats"]["steps"]==spec["budget"]==300
        assert row["instances_available"]==stored["available"] and row["reachable_instances_at_start"]==stored["reachable"]>=2
        assert row["already_at_start"]==stored["local"]["already_at_start"]==stored["previous"]["already_at_start"]
        assert row["distinct_approached"]==len(row["first_steps"])<=row["approached_half_m"]<=row["approached_one_m"]<=row["instances_available"]
        assert row["quota_met"]==(row["distinct_approached"]>=2)
        log=stage/"private_events"/f"{i}.jsonl"
        events=lines(log)
        diag=reconcile(row,events)
        slug=sha256(str(i).encode()).hexdigest()[:16]
        receipts=verify_gate_receipts(stage/"private_events"/f"gpt_{slug}.jsonl",events)
        assert diag["gpt_modes"]==row["upper_decisions"]["gpt_mode"]
        diag["gpt_evidence"]=row["upper_decisions"]["gpt_evidence"]
        diag["gpt_cue"]=row["upper_decisions"]["gpt_cue"]
        # A nonempty proposal is an opportunity, not proof that the selected view was reached.
        diag["phases_025m"]={name:thresholds["0.25"] for name,thresholds in row["distance_by_phase"].items()}
        result=dict(scene=spec["scene"],episode=spec["id"],goal=spec["goal"],
                    gate_diagnostics=diag,receipts=receipts,accepted_result_sha256=digest(op))
        for arm in ARMS:
            recorded=row if arm=="gpt" else stored[arm]
            result[arm]={m:recorded[m] for m in METRICS}
            result[arm]["first_steps"]=recorded["first_steps"]
        summaries.append(result)
    api_events=Counter()
    usage=Counter()
    for r in summaries:
        api_events.update(r["receipts"]["events"])
        usage.update(r["receipts"]["usage"])
    report=dict(panel_sha256=PANEL_HASH,model=MODEL,gate_lock_sha256=digest(seal),
                saved_complete_episodes=len(summaries),action_budget_per_arm=2400,
                provider_receipt_events=dict(api_events),reported_usage=dict(usage),
                totals=totals(summaries),by_house=summaries,
                paired={a:{m:paired(summaries,a,m) for m in ("quota_met","distinct_approached")}
                        for a in ("previous","local")},
                gpt_diagnostics={f:aggregate_counters(summaries,f) for f in
                                 ("gpt_modes","gpt_evidence","gpt_cue","region_agreement","evidence_by_mode","execution_status",
                                  "physical_actions","first_approach_phase","phases_025m")},
                previous_action_budget=prior["action_budget"])
    check_public(report)
    write_atomic(stage/"accepted.json",report)
    print(json.dumps(report["totals"]))


if __name__=="__main__":
    main()
