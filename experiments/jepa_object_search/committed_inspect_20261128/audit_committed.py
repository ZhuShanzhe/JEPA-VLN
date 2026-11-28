"""Accept only complete physical and paired provider logs; export aggregate approach diagnostics."""
import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = "gpt_inspect_only_20261128"
EARLIER = "gpt_region_20261128"
EXPECTED_PRIOR = "f18a712effe4fa2490419c717edda180de792071d66cbbd86c632f19614fd5f4"
PANEL = "9d3167d378f2f31fcd7df78d95e222fe150028bcfff739fea0f9334015e69674"
MODEL = "gpt-4o-2024-11-20"
METRICS = ("distinct_approached","approached_half_m","approached_one_m","already_at_start","quota_met")
ARMS = ("previous","local","gpt","inspect_only","committed")


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def activity_counter(rows, field):
    result = Counter()
    for row in rows:
        result.update(row[field])
    return dict(result)


def pair(rows, other, metric):
    diffs=[int(r["committed"][metric])-int(r[other][metric]) for r in rows]
    w, l = sum(d>0 for d in diffs), sum(d<0 for d in diffs)
    n = w+l
    p = min(1., 2*sum(math.comb(n,i) for i in range(min(w,l)+1))/2**n) if n else 1.
    return dict(wins=w,ties=len(rows)-n,losses=l,delta=sum(diffs),exact_two_sided_sign_p=p)


def validate_receipts(old, oldpath, newpath, decisions):
    prefix = oldpath.read_bytes()
    blob = newpath.read_bytes()
    assert blob.startswith(prefix)
    original = lines(oldpath)
    previous_success = {e["hash"]:e["decision"] for e in original if e["event"]=="success"}
    success = previous_success.copy()
    new_entries = [json.loads(line) for line in blob[len(prefix):].splitlines()]
    paid = Counter()
    usage = Counter()
    fees = []
    delivered = []
    for e in new_entries:
        kind, key = e["event"], e["hash"]
        paid[kind] += 1
        if kind in ("success","invalid_decision"):
            assert e["model"]==MODEL
            for k in ("input_tokens","output_tokens","total_tokens"):
                if e.get(k) is not None:
                    usage[k] += int(e[k])
            if e.get("provider_reported_cost") is not None:
                fees.append(float(e["provider_reported_cost"]))
        if kind == "success":
            assert key not in previous_success
            success[key]=e["decision"]
        if kind in ("cache","success"):
            assert e["decision"] == success[key]
            delivered.append((key,kind))
    n = sum(bool(e.get("gpt_called")) for e in decisions)
    assert len(delivered)>=n
    old.verify_gate_receipts(newpath,decisions)
    sources = Counter()
    for key,kind in delivered[-n:] if n else []:
        sources["identical_input_prior_response" if key in previous_success else
                ("identical_input_current_cache" if kind=="cache" else "fresh_response")] += 1
    return dict(new_journal_events=dict(paid),new_usage=dict(usage),final_response_sources=dict(sources),
                cost_entries=len(fees),reported_cost_subtotal=sum(fees) if fees else None)


def trajectory(trace):
    assert len(trace)==301 and all(e["step"]==i for i,e in enumerate(trace))
    assert trace[0]["action"]=="start"
    for i, t in enumerate(trace):
        assert len(t["xyz"])==3 and all(math.isfinite(x) for x in t["xyz"])
        assert len(t["quat_xyzw"])==4 and abs(sum(x*x for x in t["quat_xyzw"])-1)<1e-4
        assert math.isfinite(t["yaw"]) and type(t["collision"]) is bool
        if i:
            assert t["action"] in ("move_forward","turn_left","turn_right")
            if t["action"] in ("turn_left","turn_right"):
                assert math.dist(t["xyz"], trace[i-1]["xyz"])<1e-5


def local_events(row,events):
    budgets = {}
    closed = {}
    counts = Counter()
    reasons = Counter()
    for event in events:
        if event.get("gpt_called"):
            assert event["old_selected_region_id"] == event["executed_region_id"]
            assert event["old_selected_region_key"] == event["region_key"] and event["region_key"]
            if event["gpt_mode"] == "inspect":
                counts["inspect_requests"] += 1
                if event["sites"] > 0:
                    counts["inspect_requests_with_site"] += 1
            if event.get("commit_previous_id") is not None:
                if event["gpt_mode"]=="explore" and event["action_source"]=="local":
                    counts["explore_advice_during_retained_local_segment"] += 1
        release = event if event.get("event")=="arrived" else event.get("released")
        if release:
            gid = release["goal_id"]
            assert gid in budgets and gid not in closed
            closed[gid] = release["event"]
            reasons[release["event"]] += 1
        if event.get("action_source") != "local":
            continue
        gid = event["goal_id"]
        steps = event.get("last_step",event["step"])-event["step"]
        assert 0 <= steps <= event["commit_actions_left"] <= 24
        if event["event"] == "acquired":
            assert gid not in budgets and event["gpt_called"] and event["gpt_mode"]=="inspect"
            budgets[gid] = dict(start=event["step"], local_actions=0, segments=0)
        assert gid in budgets and gid not in closed
        assert event["commit_elapsed"] == event["step"]-budgets[gid]["start"]
        assert event["commit_elapsed"] + steps <= 24
        budgets[gid]["local_actions"] += steps
        budgets[gid]["segments"] += 1
    counts["acquired"] = len(budgets)
    counts["arrived"] = reasons["arrived"]
    counts["open_at_end"] = len(budgets)-len(closed)
    counts["multisegment"] = sum(v["segments"]>1 for v in budgets.values())
    counts["local_actions"] = sum(v["local_actions"] for v in budgets.values())
    assert counts["acquired"] == row["stats"].get("acquired",0)
    assert counts["local_actions"] == row["stats"].get("local_steps",0)
    assert reasons == Counter(row["counts"]["releases"])
    assert not reasons["gpt_resume_exploration"] and not reasons["gpt_new_visual_anchor"]
    return dict(counts), dict(reasons)


def compare_prefix(new,old):
    def signatures(events):
        fields = ("step","last_step","action_source","actions","contact","goal_id","site_xy","site_yaw","region_xy")
        return [(e,tuple(e.get(f) for f in fields)) for e in events if "action_source" in e]
    a,b = signatures(new),signatures(old)
    count=0
    for (_,x),(_,y) in zip(a,b):
        if x!=y:
            break
        count+=1
    return dict(equal_segment_prefix=count,complete_segment_sequences_identical=count==len(a)==len(b),
                first_changed_new_segment_after_step=(a[count][0]["step"] if count<len(a) else None))


def public_checks(report):
    rows = report["by_house"]
    assert len(rows)==len({r["scene"].split("_")[0] for r in rows})==8
    assert report["previous_report"]==EXPECTED_PRIOR and report["panel"]==PANEL
    assert report["budget_per_house"]==300 and report["model"]==MODEL
    for arm in ARMS:
        assert report["totals"][arm]=={k:sum(r[arm][k] for r in rows) for k in METRICS}
    assert report["paired"]=={a:{k:pair(rows,a,k) for k in ("distinct_approached","quota_met")} for a in ARMS[:-1]}
    for field in ("mechanisms","release_reasons","first_approach_phase","physical_actions",
                  "response_sources","new_usage","new_journal_events","phase_025m","phase_05m"):
        assert report[field] == activity_counter(rows,field)
    assert sum(report["physical_actions"].values())==2400
    assert sum(report["first_approach_phase"].values())==report["totals"]["committed"]["distinct_approached"]
    assert report["physical_actions"]["local"]==report["mechanisms"]["local_actions"]
    assert report["cost_entries"]==sum(r["cost_entries"] for r in rows)
    assert report["reported_cost_subtotal"]==(sum(r["reported_cost_subtotal"] or 0 for r in rows)
                                             if report["cost_entries"] else None)


def audit(here):
    root = here.parent
    baseline = root/PRIOR
    gate = root/EARLIER
    family = root/"local_target_20261128/eval_family"
    assert digest(baseline/"accepted.json")==EXPECTED_PRIOR
    previous = read(baseline/"accepted.json")
    assert digest(family/"panel.json")==PANEL and digest(gate/"audit.py")=="f86cc533b5255d03defac39e149f4303edd183fd01ec8aadcd1ef92a244f7b19"
    loader = importlib.util.spec_from_file_location("frozen_gpt_audit",gate/"audit.py")
    old = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(old)
    panel = read(family/"panel.json")
    lock = here/"gate_lock.json"
    for path in (lock, baseline/"gate_lock.json",gate/"gate_lock.json",family/"local_lock.json"):
        for rel,h in read(path).items():
            assert (root/rel).resolve().is_relative_to(root) and digest(root/rel)==h,rel
    rows = []
    for i,spec in enumerate(panel):
        control = previous["by_house"][i]
        result = here/"runs"/f"{i}_committed.json"
        row = read(result)
        oldresult = baseline/"runs"/f"{i}_inspect.json"
        assert digest(oldresult)==control["verified_result_sha256"]
        control_result=read(oldresult)
        fields=("scene","episode","goal")
        expected=(spec["scene"],spec["id"],spec["goal"])
        assert tuple(row[k] for k in fields)==tuple(control[k] for k in fields)==expected
        assert row["frozen_sha256"]==digest(lock) and row["arm"]=="committed"
        assert row["steps"]==row["stats"]["steps"]==spec["budget"]==300
        assert row["reachable_instances_at_start"]==control_result["reachable_instances_at_start"]>=2
        assert row["instances_available"]==control_result["instances_available"]
        assert row["already_at_start"]==0
        assert len(row["first_steps"])==row["distinct_approached"]<=row["approached_half_m"]<=row["approached_one_m"]<=row["instances_available"]
        assert bool(row["quota_met"])==(row["distinct_approached"]>=2)
        current = here/"private_events"
        log,trace = current/f"{i}.jsonl", current/f"{i}.poses.jsonl"
        assert digest(trace)==row["private_trace_sha256"] and digest(log)==row["private_events_sha256"]
        trajectory(lines(trace))
        events=lines(log)
        diag=old.reconcile(row,events)
        assert diag["gpt_modes"]==row["upper_decisions"]["gpt_mode"]
        mechanisms, reasons=local_events(row,events)
        slug=sha256(str(i).encode()).hexdigest()[:16]
        name=f"gpt_{slug}.jsonl"
        receipts=validate_receipts(old,baseline/"private_events"/name,current/name,events)
        assert sum(receipts["final_response_sources"].values()) == sum(diag["gpt_modes"].values())
        match=compare_prefix(events,lines(baseline/"private_events"/f"{i}.jsonl"))
        if match["complete_segment_sequences_identical"]:
            assert all(row[k]==control_result[k] for k in METRICS + ("first_steps",))
        record={k:row[k] for k in fields}
        record.update(mechanisms=mechanisms,release_reasons=reasons,first_approach_phase=diag["first_approach_phase"],
                      physical_actions=diag["physical_actions"],response_sources=receipts["final_response_sources"],
                      new_usage=receipts["new_usage"],new_journal_events=receipts["new_journal_events"],
                      cost_entries=receipts["cost_entries"],reported_cost_subtotal=receipts["reported_cost_subtotal"],
                      compared_to_inspect_only=match,verified_result_sha256=digest(result))
        for key,t in (("phase_025m","0.25"),("phase_05m","0.5")):
            record[key]={k:v[t] for k,v in row["distance_by_phase"].items()}
        record.update({arm:control[arm] for arm in ARMS[:-1]})
        record["committed"]={k:row[k] for k in METRICS+("first_steps",)}
        rows.append(record)
    agg=("mechanisms","release_reasons","first_approach_phase","physical_actions","response_sources","new_usage",
         "new_journal_events","phase_025m","phase_05m")
    nfees=sum(r["cost_entries"] for r in rows)
    report=dict(model=MODEL,budget_per_house=300,panel=PANEL,previous_report=EXPECTED_PRIOR,arm_lock=digest(lock),
                task="Geodesic approach to a distinct instance reference viewpoint; correct report unmeasured",
                cost_entries=nfees,reported_cost_subtotal=sum(r["reported_cost_subtotal"] or 0 for r in rows) if nfees else None,
                totals={a:{k:sum(r[a][k] for r in rows) for k in METRICS} for a in ARMS},
                paired={a:{k:pair(rows,a,k) for k in ("distinct_approached","quota_met")} for a in ARMS[:-1]},
                by_house=rows,**{k:activity_counter(rows,k) for k in agg})
    public_checks(report)
    old.write_atomic(here/"accepted.json",report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--public-only",action="store_true")
    args=parser.parse_args()
    if args.public_only:
        public_checks(read(HERE/"accepted.json"))
        print("Public report verified")
    else:
        print(json.dumps(audit(HERE)["totals"]))
