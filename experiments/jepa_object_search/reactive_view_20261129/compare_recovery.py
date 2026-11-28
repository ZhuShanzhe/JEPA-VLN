"""Counterfactual recovery head: replace only JEPA visibility by the frozen projected-area rule."""
from collections import Counter,defaultdict
import json
from pathlib import Path
from score_rollout import group_metrics

HERE=Path(__file__).resolve().parent


def main():
    results=defaultdict(lambda:defaultdict(Counter))
    houses=defaultdict(lambda:defaultdict(Counter))
    files=sorted((HERE/"private_roll").glob("*/*.json"))
    assert len(files)==180
    for path in files:
        j=json.loads(path.read_text())
        area=json.loads((HERE/"private_roll_area"/path.parent.name/path.name).read_text())
        assert area["group"]==j["group"] and area["baseline_length"]==j["baseline_length"]
        prefix=j["baseline_length"]+1
        assert area["events"][:prefix]==j["events"][:prefix]
        scores={}
        for name,row in (("area",area),("jepa",j)):
            actual=group_metrics(row,"new_confirm")
            shadow=group_metrics(row,"new_no_stop")
            t=row["new_confirmation"]
            scores[name]=dict(groups=1,confirmed=actual["correct"],saved=actual["saved"],
                incorrect=actual["invalid"],actions=actual["actions"],contacts=actual["contacts"],
                physical_opportunity=shadow["opportunity"],actions_no_stop=shadow["actions"],
                confirmed_after_legacy=int(actual["correct"] and t>row["baseline_length"]))
        diverged=any(x["action"]!=y["action"] for x,y in zip(j["events"],area["events"])) or len(j["events"])!=len(area["events"])
        a_ok=bool(scores["area"]["confirmed"]);j_ok=bool(scores["jepa"]["confirmed"])
        a_seen=bool(scores["area"]["physical_opportunity"]);j_seen=bool(scores["jepa"]["physical_opportunity"])
        for subset,keep in (("all",True),("seven_blocked",j["seven_blocked"]),("focus",j["focus"])):
            if not keep:continue
            for name,value in scores.items():results[subset][name].update(value)
            tally=dict(paired=1,route_diverged=int(diverged),same_shadow_actions=int(not diverged),
                both_confirm=int(a_ok and j_ok),neither_confirm=int(not a_ok and not j_ok),
                jepa_only_confirm=int(j_ok and not a_ok),area_only_confirm=int(a_ok and not j_ok),
                jepa_only_opportunity=int(j_seen and not a_seen),area_only_opportunity=int(a_seen and not j_seen))
            results[subset]["paired"].update(tally)
            houses[subset][j["house"]].update(tally)
    out=dict(scope="Same frozen first-leg actions, contact trigger, locally accumulated RGB-D, remaining budget, candidate paths, shortest admitted path per view, path checker and first-frame confirmation rule. Only recovery-view merit changes: classifier p_visible vs logistic of clipped geometric log projected area; each is multiplied by the same registered p_reach.",
             subsets=results,per_house={s:[v for k,v in sorted(h.items())] for s,h in houses.items()})
    (HERE/"recovery_control.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(results))


if __name__=="__main__":main()
