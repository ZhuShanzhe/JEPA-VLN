"""Freeze a second high-precision admission threshold using exclusively validation outcomes."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def split(name):
    scores,labels=[],[]
    for path in sorted((HERE/"private_checker").glob(name+"_*.npz")):
        with np.load(path) as pred,np.load(HERE/"private_data"/path.name) as data:
            for key in ("group","plan_id"):assert np.array_equal(pred[key],data[key])
            scores.extend(pred["p_reach"])
            labels.extend(data["reach"])
    return np.array(scores),np.array(labels,dtype=bool)


def row(scores,labels,t):
    accepted=scores>=t
    n=int(accepted.sum())
    tp=int((labels&accepted).sum())
    false_veto=int((labels&~accepted).sum())
    false_permit=n-tp
    return dict(n=len(labels),feasible=int(labels.sum()),blocked=int((~labels).sum()),
                threshold=float(t),accepted=n,accepted_feasible=tp,precision=tp/n if n else None,
                coverage=n/len(labels),false_veto=false_veto,false_permit=false_permit,
                false_veto_rate=false_veto/int(labels.sum()),false_permit_rate=false_permit/int((~labels).sum()))


def main():
    primary=json.loads((HERE/"checker_results.json").read_text())["frozen_threshold"]
    s,y=split("val")
    # All attainable states at distinct scores in descending order, including ties.
    order=np.argsort(-s,kind="stable")
    sorted_scores=s[order]
    cuts=np.flatnonzero(np.r_[np.diff(sorted_scores)!=0,True])
    totals=cuts+1
    precision=np.cumsum(y[order])[cuts]/totals
    permitted=np.flatnonzero((totals>=30)&(precision>=.85))
    conservative=None if not len(permitted) else float(sorted_scores[cuts[permitted[-1]]])
    thresholds={"advisory":primary,"conservative":conservative}
    # Save a concise calibration curve plus the exact operating points and maximum attainable precision.
    tcurve=np.unique(np.r_[np.linspace(0,1,51),[q for q in thresholds.values() if q is not None]])
    result=dict(protocol="One calibration split; require >=85% observed passability among at least 30 approved calibration proposals, maximize approval count. Test is consulted only after the threshold is fixed.",
                thresholds=thresholds,status="available" if conservative is not None else "constraint_not_attained",
                val_curve=[row(s,y,t) for t in tcurve],val_distinct_scores=len(cuts),
                maximum_precision_with_at_least_30=float(precision[totals>=30].max()))
    for kind,t in thresholds.items():
        if t is None:continue
        result[kind]={"val":row(s,y,t),"test":row(*split("test"),t)}
    (HERE/"checker_operating_points.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(result["status"],thresholds)
    for k in thresholds:
        print(k,result.get(k))

if __name__=="__main__":main()
