"""Transfer test for untemplated 12–108-action paths. Repeated target cues per body path are counted once."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from checker import PathChecker
from mixed_checker import MixedPathChecker
from path_geometry import path_features,from_depth,NAMES

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/"pose_horizon_20261128"


def metrics(y,p,t):
    take=p>=t
    f=(~y)&take
    b=y&~take
    return dict(n=len(y),reachable=int(y.sum()),blocked=int((~y).sum()),
        accepted=int(take.sum()),blocked_allowed=int(f.sum()),reachable_rejected=int(b.sum()),
        brier=float(np.mean((y.astype(float)-p)**2)),
        auroc=float(roc_auc_score(y,p)) if np.unique(y).size>1 else None)


def main():
    model=PathChecker.load(HERE/"private_checker/checker.joblib")
    mixed=MixedPathChecker.load(HERE/"private_checker_mixed/checker.joblib")
    t=float(model.models["threshold"])
    old_t=json.loads((HERE/"checker_results.json").read_text())["previous_matched_threshold"]
    records=[]
    with np.load(PRIOR/"private_train/test_long_cue_joint_predictions.npz") as pred:
        for name in sorted(set(pred["source_file"])):
            with np.load(PRIOR/"private_long_cue"/name) as z,np.load(HERE/"private_transfer_depth"/name) as images:
                depths={org:from_depth(d) for org,d in zip(images["origin"],images["depth"])}
                rows=pred["source_row"][pred["source_file"]==name]
                prev=dict(zip(rows,pred["p_reach"][pred["source_file"]==name]))
                keys={}
                for i in np.flatnonzero(z["native_mask"]):
                    key=(str(z["origin"][i]),int(z["plan_id"][i]),int(z["horizon"][i]))
                    keys.setdefault(key,[]).append(i)
                x=np.zeros((len(keys),108,len(NAMES)),np.float32)
                acts=[];truth=[];old=[];h=[];flags=[]
                for j,((org,pid,n),ids) in enumerate(keys.items()):
                    i=ids[0]
                    a=z["action"][i]
                    assert np.all(z["action"][ids]==a) and np.unique(z["reach"][ids]).size==1
                    x[j,:n]=path_features(tuple(a[:n]),depths[org])
                    acts.append(a);truth.append(z["reach"][i]);h.append(n)
                    old.append(np.mean([prev[k] for k in ids]))
                    flags.append(bool(z["safe_mask"][i]))
                out=model.predict(np.array(acts),x)
                extended=mixed.predict(np.array(acts),x)
                records.extend(zip(h,truth,out["p_reach"],old,flags,extended["p_reach"],extended["threshold"],
                                   extended["screening"]=="admit",extended["screening"]=="block"))
    h,y,p,q,g,pm,tm,admitted,denied=np.asarray(records).T
    y=y.astype(bool)
    report=dict(scope="Existing natural action trees; native camera; repeated target crops for one physical route deduplicated; previous target-conditioned reach estimates averaged over its cues. Current RGB-D source reconstructed and source JPEG verified. Frozen checker thresholds used without tuning.",
        learned_threshold=t,previous_calibrated_on_new_val=old_t,subsets={})
    for k,m in [("all",np.ones(len(h),bool))]+[(str(int(n)),h==n) for n in np.unique(h)]:
        report["subsets"][k]=dict(checker=metrics(y[m],p[m],t),previous=metrics(y[m],q[m],.5),
            previous_calibrated=metrics(y[m],q[m],old_t),geometry=metrics(y[m],g[m],.5))
        admit=admitted[m].astype(bool)
        block=denied[m].astype(bool)
        report["subsets"][k].update(mixed=metrics(y[m],pm[m],tm[m]),mixed_selective={
            "admitted":int(admit.sum()),"admitted_feasible":int(y[m][admit].sum()),
            "denied":int(block.sum()),"denied_actually_blocked":int((~y[m][block]).sum()),
            "deferred":int((~admit&~block).sum())})
    dst=HERE/"transfer_results.json"
    if dst.exists():
        prior=json.loads(dst.read_text())
        for subset,metrics_before in prior["subsets"].items():
            for name in ("checker","previous","previous_calibrated","geometry"):
                assert report["subsets"][subset][name]==metrics_before[name],(subset,name)
    report["mixed_operating_points"]=mixed.limits
    dst.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(report["subsets"]["all"])

if __name__=="__main__":
    with threadpool_limits(limits=2):main()
