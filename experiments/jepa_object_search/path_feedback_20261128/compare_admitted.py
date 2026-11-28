"""Compare selection rules with identical proposals, mixed checker and per-group action budget."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from train_repair import groups
from replan import RepairScorer
from blind_repair import screen_first

HERE=Path(__file__).resolve().parent


def main():
    import torch
    torch.set_num_threads(2)
    scorer=RepairScorer(HERE/"private_repair"/"with_hint.pt")
    summaries=defaultdict(Counter)
    per_house=defaultdict(lambda:defaultdict(Counter))
    cache={}
    for house,_,s,reached,visible,ids in groups("blind",HERE/"private_blind_paired",
            HERE/"private_blind_checker_mixed",HERE/"private_blind_repair"):
        if house not in cache:
            with np.load(HERE/"private_blind_paired"/(house+".npz"),allow_pickle=False) as z:
                cache[house]=z["first_blocked"].copy()
        first=cache[house][ids]
        actual=np.where(first>0,first,s.length)
        assert ((actual>0)&(actual<=s.length)).all()
        allowed=np.flatnonzero(s.decision=="accept")
        baseline=dict(straight=0,old_jepa=int(np.argmax(s.old_joint)))
        choices=dict(shortest=None,highest_reach=None,reach_times_jepa_visible=None)
        if len(allowed):
            # Candidate ID is the fixed tie-break shared by all rules.
            choices.update(shortest=int(allowed[np.argmin(s.length[allowed])]),
                           highest_reach=int(allowed[np.argmax(s.reach[allowed])]),
                           reach_times_jepa_visible=int(allowed[np.argmax((s.reach*s.visible_if_reached)[allowed])]))
        triggers={}
        for count in (3,7):
            arm=f"first_{count}_shortest_then_repair"
            admitted=allowed[allowed<count]
            triggers[arm]=not len(admitted)
            if len(admitted):
                choices[arm]=int(admitted[np.argmin(s.length[admitted])])
            else:
                # There are no unseen templates when all seven are already rejected.
                selected,triggered=screen_first(s,scorer,existing=tuple(range(count)))
                assert triggered
                choices[arm]=selected
        for arm,i in (baseline|choices).items():
            row=dict(groups=1,issued=int(i is not None),reobserve=int(i is None),
                     reached=int(i is not None and reached[i]),clear=int(i is not None and reached[i] and visible[i]),
                     blocked=int(i is not None and not reached[i]),nominal_actions=int(s.length[i]) if i is not None else 0,
                     actions_until_stop=int(actual[i]) if i is not None else 0)
            if arm in triggers:
                trigger=triggers[arm]
                row.update(repair_triggered=int(trigger),repair_issued=int(trigger and i is not None),
                           repair_reached=int(trigger and i is not None and reached[i]),
                           repair_clear=int(trigger and i is not None and reached[i] and visible[i]))
            per_house[house][arm].update(row);summaries[arm].update(row)
    assert len(per_house)==6 and all(summary["groups"]==620 for summary in summaries.values())
    pairs={}
    rng=np.random.default_rng(1128)
    bootstrap=rng.integers(6,size=(10000,6))
    comparisons=[("reach_times_jepa_visible",arm) for arm in summaries if arm!="reach_times_jepa_visible"]
    comparisons.extend((("first_3_shortest_then_repair","first_7_shortest_then_repair"),
                        ("first_3_shortest_then_repair","straight"),("first_7_shortest_then_repair","straight")))
    for ref,arm in comparisons:
        delta=np.array([(v[ref]["clear"]-v[arm]["clear"])/v[ref]["groups"] for _,v in sorted(per_house.items())])
        pairs[ref+"_minus_"+arm]=dict(house_mean=float(delta.mean()),house_deltas=delta.tolist(),
                    interval95=np.quantile(delta[bootstrap].mean(1),[.025,.975]).tolist())
    report=dict(houses=len(per_house),counts=summaries,clear_rate_paired_house_bootstrap=pairs,
                protocol="All 620 groups share seven physical equal-endpoint plans with a common within-group upper action limit. Screened policies use the same mixed checker. ID breaks ties. First-3 starts from direct and both narrow offsets; first-7 starts with all templates. Both choose shortest among screened-in initial plans. On complete rejection the frozen feedback head can offer up to two previously unreviewed plans; when all seven have been reviewed, the response requests observation. Straight and old JEPA may issue rejected routes. Actual cost is stop-on-first-contact with rejected fallback to new observation having zero issued actions. Observation requests counted separately; no cost for a subsequent scan or full-episode claim.")
    (HERE/"admitted_fair_comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"counts":summaries,"pair":pairs},ensure_ascii=False))

if __name__=="__main__":main()
