"""Evaluate two candidate decisions with both possible first choices physically recorded."""
import json
from pathlib import Path
import sys
import numpy as np


def load(p):
    return json.loads(p.read_bytes())


def main(d):
    choices=[load(p/"blind_predictions.json") for p in (d,d/"reverse")]
    v10=load(d/"test.predictions.json")["episodes"]
    rows={};v10_stats={};ranks={}
    max_prefix=0.;different_frames=[]
    for sid,left in choices[0].items():
        right=choices[1][sid]
        a=[load(p/"streams/test"/(sid+".actor.json"))["history"] for p in (d,d/"reverse")]
        mismatch=[x[0] for x,y in zip(*a) if x!=y]
        if not mismatch or mismatch[0]<=180:raise ValueError(f"Histories not matched: {sid}")
        different_frames.append(mismatch[0])
        gt=load(d/"streams/test"/(sid+".truth.json"));other=load(d/"reverse/streams/test"/(sid+".truth.json"))
        layout="moved" if gt["relocated"] else "stationary"
        assert [s["query_routes"] for s in gt["labels"]]==[s["query_routes"] for s in other["labels"]]
        for i in range(5):
            for arm,v in left[i]["scores"].items():
                max_prefix=max(max_prefix,float(np.max(np.abs(np.array(v)-right[i]["scores"][arm]))))
        for q in (0,1):
            truth=np.argmax(gt["labels"][4]["query_routes"][q])
            controls={"memory_first_then_other":"jepa_plus_update","static_first_then_other":"jepa_prior_only",
                      "fixed_left_then_other":None}
            for arm in [*left[4]["scores"],*controls]:
                source=controls.get(arm,arm)
                first=0 if source is None else int(np.argmax(np.array(left[4]["scores"][source])[:,q]))
                # If this physical opportunity is missed, use that actual stream's
                # new image history and count the 36 actions required to get back.
                actual=choices[first][sid]
                second=1-first if arm in controls else int(np.argmax(np.array(actual[5]["scores"][arm])[:,q]))
                win=(first==truth,first==truth or second==truth)
                rec=[*map(float,win),24. if win[0] else 84.,float(first),float(second!=first)]
                for g in ("all",gt["cue"],gt["house"]):
                    rows.setdefault((layout,g,arm),[]).append(rec)
        for i,s in enumerate(gt["labels"]):
            valid=np.array(s["query_routes"])
            for arm in v10[sid][0][i]["scores"]:
                values=np.array([row[i]["scores"][arm] for row in v10[sid]])
                diff=(values*(2*valid-1)).sum(1)
                acc=(diff>1e-7)+.5*(np.abs(diff)<=1e-7)
                v10_stats.setdefault((layout,s["phase"],arm),[]).extend(acc)
            for idx,order in enumerate(("left_first","right_first")):
                for arm,values in choices[idx][sid][i]["scores"].items():
                    diff=(np.array(values).T*(2*valid-1)).sum(1)
                    acc=(diff>1e-7)+.5*(np.abs(diff)<=1e-7)
                    phase="revisited_right" if idx==1 and i==5 else s["phase"]
                    ranks.setdefault((layout,order,phase,arm),[]).extend(acc)
    assert max_prefix<1e-8 and len(choices[0])==72
    policy={}
    for (cond,g,arm),sample in rows.items():
        col=np.array(sample)
        fail_first=col[:,0]==0
        policy.setdefault(cond+"/"+g,{})[arm]=dict(n=len(col),
            first_success_at_24=int(col[:,0].sum()),success_within_84=int(col[:,1].sum()),
            mean_capped_actions=float(col[:,2].mean()),chose_right_first=int(col[:,3].sum()),
            failed_first_then_other=int(col[fail_first,4].sum()),failed_first=int(fail_first.sum()))
    stats={}
    for (cond,phase,arm),acc in v10_stats.items():stats.setdefault(cond+"/"+phase,{})[arm]=float(np.mean(acc))
    fixed={}
    for (cond,order,phase,arm),acc in ranks.items():fixed.setdefault(cond+"/"+order+"/"+phase,{})[arm]=float(np.mean(acc))
    result=dict(same_histories=72,first_divergent_action_range=[min(different_frames),max(different_frames)],
        prechoice_score_disagreement=max_prefix,fixed_history_actions=180,
        decision_protocol="argmax per goal; ties choose left; 24 moves to inspect; 36 moves to return if unsuccessful; second 24-action budget. Success means actual visible, reachable target opportunity, not an agent-generated object report.",
        maximum_additional_actions=84,policy=policy,v10_fixed_path=stats,both_first_revisit_orders=fixed)
    (d/"policy_summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in policy.items() if k.endswith("/all")},indent=2))


if __name__=="__main__":main(Path(sys.argv[1]).resolve())
