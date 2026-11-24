"""Audit chronology against same-observation static route choices after execution."""
import inspect
import json
from pathlib import Path
import sys


def count_progress(case,ep,near,rows,selects,active,decisions,reports):
    ever=set()
    first_near={}
    for t,ids in enumerate(near):
        for obj in ids-ever:first_near[obj]=t
        ever|=ids
    first_report={}
    for t,ids in sorted(reports.items()):
        for obj in ids:
            assert obj not in first_report
            first_report[obj]=t
    assert len(first_report)==int(ep["found_count"])
    quota=int(case["quota"])
    quota_at=None
    for t in sorted(set(first_report.values())):
        if sum(v<=t for v in first_report.values())>=quota:
            quota_at=t;break
    assert (quota_at is not None)==bool(ep["success"])
    events=[]
    actions=[x["action"] for x in rows]
    for sel,activation in zip(selects,active):
        if "scores" not in sel or not activation["applied"]:continue
        name=sel["frozen_selected"]
        if name==sel["selected"]:continue
        s=sel["step"]
        plan=sel["selected_nominal_actions"]
        alt=(sel["original_nominal_actions"] if name==sel["original"] else
             [a+1 for a in sel["supported_plans"][name]])
        j=next(i for i,(a,b) in enumerate(zip(plan,alt)) if a!=b)
        fork=s+j
        future_native=[z["step"] for z in ep["paper"]["agent_trace"]["decisions"] if z["step"]>s]
        ends=[z for z in decisions if z["event"]=="end" and z["start"]==s]
        end=min([s+len(plan),len(rows),*future_native])
        if ends:
            end=min(end,ends[0]["step"]+int(ends[0]["reason"]=="episode_budget"))
        if fork>=end or actions[s:fork+1]!=plan[:j+1]:continue
        if sel["selected"].startswith("motion:"):
            dispatched=[e for e in decisions if e["event"]=="command" and e["start"]==s]
            if len(dispatched)<=j:continue
        new={k for k,t in first_near.items() if fork<t<=end}
        eventual=set(first_report)&new
        events.append(dict(decision_step=s,first_executed_divergence_step=fork,control_window_end=end,
                           first_newly_eligible_in_window=min((first_near[k] for k in new),default=None),
                           new_in_window=len(new),eventually_newly_reported=len(eventual),
                           first_eventual_report=min((first_report[k] for k in eventual),default=None)))
    start=events[0]["first_executed_divergence_step"] if events else None
    new_after={k for k,t in first_near.items() if start is not None and t>start}
    before=sum(t<=start for t in first_report.values()) if start is not None else None
    return dict(ordinal=case["ordinal"],raw_unique_credited=len(first_report),quota=quota,
                capped_fraction=min(len(first_report),quota)/quota,official_recall=ep["recall"],
                first_executed_online_static_divergence=start,unique_credited_by_first_fork=before,
                first_ever_eligible_step=min(first_near.values(),default=None),
                first_previously_unreached_instance_after_first_fork=min((first_near[k] for k in new_after),default=None),
                first_report_with_new_credit=min(first_report.values(),default=None),quota_reached_step=quota_at,
                raw_unique_by_report_step=[[t,sum(x==t for x in first_report.values())] for t in sorted(set(first_report.values()))],
                executed_divergences=events)


def main(d):
    sys.path.insert(0,str(d))
    import motion_audit as native
    source=inspect.getsource(native.audit)
    cut=source.index("    result_events=[]")
    source=source[:cut]+"    return count_progress(case,ep,near,rows,selects,activation,decisions,reports)\n"
    ns={**vars(native),"count_progress":count_progress}
    exec(compile(source,"<post-run counterfactual chronology>","exec"),ns)
    rows=[]
    for r in native.read(d/"panel.json")["rows"]:
        store=d/f"chronology_{r['ordinal']}.json"
        if not store.is_file():
            if not (d/"replay"/(r["episode_id"]+".json")).exists():continue
            study=ns["audit"](r)
            store.write_text(json.dumps(study,indent=2))
        rows.append(native.read(store))
    if len(rows)!=5:return
    out=dict(cases=rows)
    f=d/"chronology.json"
    if f.exists():assert native.read(f)==out
    else:f.write_text(json.dumps(out,indent=2))
    print("replayed chronological route attribution",len(rows))


if __name__=="__main__":main(Path(sys.argv[1]))
