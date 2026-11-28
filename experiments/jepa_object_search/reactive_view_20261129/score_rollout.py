"""Paired comparisons from physically executed controls. No simulator labels available to either policy."""
from collections import Counter,defaultdict
from pathlib import Path
import argparse,json
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"new_view_20261129"


def group_metrics(row,arm):
    old=arm.startswith("old")
    confirm=arm.endswith("_confirm")
    base=row["baseline_length"]
    cap=base if old else len(row["events"])-1
    first=row["old_confirmation" if old else "new_confirmation"] if confirm else None
    stop=first if first is not None else cap
    events=row["events"][:stop+1]
    valid=first is not None and events[-1]["correct"]
    wrong=first is not None and not valid
    saved=first is not None
    any_clear=any(e["px"]>=369 for e in events)
    return Counter(groups=1,actions=stop,opportunity=int(any_clear),saved=int(saved),
                   correct=int(valid),invalid=int(wrong),missed_seen=int(not saved and any_clear),
                   below_pixel_threshold=int(wrong and events[-1]["px"]<369),
                   spatial_mismatch_when_clear=int(wrong and events[-1]["px"]>=369),
                   contacts=sum(e["bump"] for e in events),full_budget=int(stop==row["budget"]),
                   terminal_frame_clear=int(events[-1]["px"]>=369))


def main(split="private_roll"):
    design=json.loads((OLD/"view_design.json").read_text())["cohorts"]["blind"]
    panel=[]
    for h in design:
        files=sorted((HERE/split/h).glob("*.json"))
        assert len(files)==60,(h,len(files))
        panel.extend(json.loads(f.read_text()) for f in files)
    assert len({(r["house"],r["source"]) for r in panel})==180
    def audit(records):
        arms={a:sum((group_metrics(r,a) for r in records),Counter()) for a in
              ("old_no_stop","new_no_stop","old_confirm","new_confirm")}
        pairs=defaultdict(Counter)
        cases=Counter()
        cuts={s:Counter() for s in (6,10,16,22,30)}
        per_house=defaultdict(Counter)
        diag=Counter()
        for r in records:
            b=r["baseline_length"]; ev=r["events"]
            assert ev[0]["action"]==0 and len(ev)<=r["budget"]+1 and len(ev)>=b+1
            assert all(i==e["t"] for i,e in enumerate(ev))
            old_gain=any(e["px"]>=369 for e in ev[:b+1])
            new_gain=any(e["px"]>=369 for e in ev)
            success=[]
            for a in ("old","new"):
                t=r[a+"_confirmation"]
                success.append(t is not None and ev[t]["correct"])
                if t is not None:assert ev[t]["accept"] and all(not e["accept"] for e in ev[:t])
            for metric,(before,after) in (("any_opportunity",(old_gain,new_gain)),
                                           ("first_confirmation_correct",success)):
                pairs[metric].update(win=int(after and not before),loss=int(before and not after))
                per_house[r["house"]][metric]+=int(after)-int(before)
            if success[1] and not success[0]:cases["correct_additional_from_extension"]+=1
            old_stop=r["old_confirmation"];new_stop=r["new_confirmation"]
            if success[1]:
                cases["correct_on_legacy_prefix" if new_stop<=b else "correct_after_extension"]+=1
            elif new_stop is not None:
                cases["wrong_first_confirmation"]+=1
            elif new_gain:cases["true_visible_but_abstained"]+=1
            else:cases["never_clear_within_budget"]+=1
            for cutoff,c in cuts.items():
                old_frames=ev[:min(b,cutoff)+1];new_frames=ev[:min(len(ev)-1,cutoff)+1]
                c["old_ever"]+=any(e["px"]>=369 for e in old_frames)
                c["new_ever"]+=any(e["px"]>=369 for e in new_frames)
                c["old_confirmed"]+=int(success[0] and old_stop<=cutoff)
                c["new_confirmed"]+=int(success[1] and new_stop<=cutoff)
            actual_replans=[op for op in r["replans"] if new_stop is None or op["t"]<new_stop]
            for op in actual_replans:
                d=op["details"]
                diag["decisions"]+=1
                diag["contact_trigger"]+=op["trigger"]=="contact_or_stall"
                diag["distance_over_1_5m"]+=float(d.get("local_distance_m",0))>1.5
                diag["direction_over_90deg"]+=abs(float(d.get("relative_bearing_deg",0)))>90
                diag[d["reason"]]+=1
            if actual_replans:diag["requests_replanned"]+=1
        return dict(arms=arms,changes=pairs,attribution=cases,cumulative=cuts,
                    signed_gains_per_house=[dict(v) for _,v in sorted(per_house.items())],
                    recovery_actions_before_actual_stop=diag)
    subsets={"all":panel,"seven_blocked":[r for r in panel if r["seven_blocked"]],
             "both_initial_unclear":[r for r in panel if max(e["px"] for e in r["events"][:3])<369],
             "focus":[r for r in panel if r["focus"]]}
    assert len(subsets["focus"])==94 and len(subsets["seven_blocked"])==133
    out={k:audit(v) for k,v in subsets.items()}
    assert out["focus"]["arms"]["old_no_stop"]["opportunity"]==40
    assert out["all"]["arms"]["old_no_stop"]["opportunity"]==105
    assert out["all"]["arms"]["old_no_stop"]["actions"]==1550
    if split=="private_roll":
        (HERE/"closed_loop_metrics.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({s:{"arms":v["arms"],"change":v["changes"],"attribution":v["attribution"]}
                      for s,v in out.items() if s in ("focus","all")}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--split",default="private_roll")
    main(p.parse_args().split)
