"""Fit on training buildings, freeze operating points on separate calibration buildings, test once."""
from pathlib import Path
from collections import Counter
import json,os,sys
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits

from checker import PathChecker,check_inputs,per_step,summarize
from path_geometry import NAMES

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"pose_horizon_20261128"
PRIVATE=HERE/"private_checker"


def cutoff(y,p):
    # Operating point: maximize acceptance of genuinely passable trajectories subject to allowing
    # at most 15 percent of truly blocked calibration trajectories. Never fit on test houses.
    scores=[]
    for t in np.unique(np.r_[0.,p,np.nextafter(1.,2.)]):
        v=p>=t
        bad=float(v[~y].mean())
        if bad<=.15:
            scores.append((float(v[y].mean()),-bad,float(t)))
    return max(scores)[-1]


def classification(y,s,t):
    if not len(y):return {"n":0}
    accepted=s>=t
    return dict(n=len(y),feasible=int(y.sum()),blocked=int((~y).sum()),accepted=int(accepted.sum()),
                false_veto=int((~accepted&y).sum()),false_permit=int((accepted&~y).sum()),
                false_veto_rate=float((~accepted[y]).mean()) if y.any() else None,
                false_permit_rate=float(accepted[~y].mean()) if (~y).any() else None,
                auroc=float(roc_auc_score(y,s)) if np.unique(y).size>1 else None,
                ap_blocked=float(average_precision_score(~y,1-s)) if np.unique(y).size>1 else None,
                brier=float(((y-s)**2).mean()))


def get_data():
    parts={}
    for split in ("train","val","test"):
        inputs=sorted((HERE/"private_data").glob(split+"_*.npz"))
        assert len(inputs)=={"train":12,"val":6,"test":12}[split],(split,len(inputs))
        shards=[]
        for f in inputs:
            with np.load(f,allow_pickle=False) as z:
                if "path_names" in z:assert list(z["path_names"])==list(NAMES)
                shard={k:z[k] for k in ("path","action","group","plan_id","reach","safe_mask","first_blocked","context","pose")}
                n=len(shard["reach"])
                shard["file"]=np.repeat(f.name,n)
                shard["house"]=np.repeat(f.stem.split("_",1)[1],n)
                assert not n or set(z["house"])==set(shard["house"])
                shards.append(shard)
        d={k:np.concatenate([s[k] for s in shards]) for k in shards[0]}
        check_inputs(d["action"],d["path"])
        assert np.isin(d["reach"],[0,1]).all()
        d["reach"]=d["reach"].astype(bool)
        d["safe_mask"]=d["safe_mask"].astype(bool)
        parts[split]=d
    fam=[{h.split("_")[0] for h in d["house"]} for d in parts.values()]
    assert all(fam[i].isdisjoint(fam[j]) for i,j in ((0,1),(0,2),(1,2)))
    return parts


def out_of_fold_predictions(train,val):
    # Avoid teaching the downstream repair module with an upstream score fitted to that house.
    houses=np.array([h.split("_")[0] for h in train["house"]])
    folds=np.array_split(np.random.default_rng(1128).permutation(np.unique(houses)),3)
    pred={}
    for fold in folds:
        withheld=np.isin(houses,fold)
        rest={k:train[k][~withheld] for k in ("path","action","reach","first_blocked")}
        learner=PathChecker(fit(rest,val))
        rows=learner.predict(train["action"][withheld],train["path"][withheld])
        for k,x in rows.items():
            if k not in pred:pred[k]=np.empty(len(houses),x.dtype)
            pred[k][withheld]=x
    return pred


def fit(train,val):
    a,x=train["action"],train["path"]
    route=HistGradientBoostingClassifier(max_leaf_nodes=15,max_iter=110,l2_regularization=2,
                                         early_stopping=False,random_state=1128)
    route.fit(summarize(a,x),train["reach"])
    first=train["first_blocked"]
    step=np.arange(a.shape[1])[None,:]+1
    usable=(a==1)&((first[:,None]==0)|(step<=first[:,None]))
    label=(step==first[:,None])
    assert not ((first>0)&train["reach"]).any() and label[usable].any()
    local=HistGradientBoostingClassifier(max_leaf_nodes=11,max_iter=80,l2_regularization=2,
                                         early_stopping=False,random_state=1128)
    local.fit(per_step(a,x)[usable],label[usable])
    a,x=val["action"],val["path"]
    scores=route.predict_proba(summarize(a,x))[:,1]
    scores[~(a==1).any(1)]=1.
    t=cutoff(val["reach"],scores)
    return dict(route=route,local=local,threshold=t,names=NAMES)


def template_features(actions,paths):
    """Geometry of intended controls only; tests whether route-length or turn frequencies suffice."""
    out=[]
    for a,x in zip(actions,paths):
        end=int((a!=0).sum())-1
        loc=x[:end+1,:2]
        out.append([end+1,(a==1).sum(),(a==2).sum(),(a==3).sum(),*loc[-1],*loc.min(0),*loc.max(0)])
    return np.asarray(out)


def score_prior(data,network):
    p=[]
    for lo in range(0,len(data["reach"]),128):
        sl=slice(lo,lo+128)
        out=network(data["context"][sl],data["action"][sl],data["pose"][sl])
        p.append(out["p_reach"])
    return np.concatenate(p)


def bootstrap_diff(y,p,q,house):
    d=[]
    for h in np.unique(house):
        m=house==h
        if np.unique(y[m]).size<2:continue
        d.append(roc_auc_score(y[m],p[m])-roc_auc_score(y[m],q[m]))
    if not d:return {}
    draws=np.random.default_rng(1128).choice(d,(5000,len(d))).mean(1)
    return dict(informative_houses=len(d),mean_house_auroc_delta=float(np.mean(d)),
                bootstrap95=np.quantile(draws,[.025,.975]).tolist())


def main():
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(OLD))
    from observation_head import HorizonCameraPredictor
    os.umask(0o077)
    PRIVATE.mkdir(exist_ok=True)
    data=get_data()
    models=fit(data["train"],data["val"])
    template=HistGradientBoostingClassifier(max_leaf_nodes=15,max_iter=110,l2_regularization=2,
                                             early_stopping=False,random_state=1128)
    train=data["train"]
    template.fit(template_features(train["action"],train["path"]),train["reach"])
    joblib.dump(models,PRIVATE/"checker.joblib")
    checker=PathChecker.load(PRIVATE/"checker.joblib")
    old=HorizonCameraPredictor.from_checkpoints(OLD/"private_train","cuda" if torch.cuda.is_available() else "cpu")
    report=dict(training_columns=list(NAMES),frozen_threshold=models["threshold"],
                threshold_protocol="Maximize feasible acceptance at <=15% false permit on calibration buildings; ties prefer lower false permits then higher threshold.",
                scope="All model inputs use only source depth and the proposed primitives. Frozen previous model is also given source RGB and observed target crop.",
                splits={})
    prior_t=None
    template_t=None
    for split in ("val","train","test"):
        d=data[split]
        y=d["reach"]
        a,x=d["action"],d["path"]
        out=out_of_fold_predictions(d,data["val"]) if split=="train" else checker.predict(a,x)
        p=out["p_reach"]
        prev=score_prior(d,old)
        ts=template.predict_proba(template_features(a,x))[:,1]
        if split=="val":prior_t=cutoff(y,prev);template_t=cutoff(y,ts)
        moving=(a==1)
        cover=(x[:,:,5]*moving).sum(1)/np.maximum(1,moving.sum(1))
        body_seen=np.any((x[:,:,13]>.5)&moving,axis=1)
        low=(cover<.65)&~body_seen&moving.any(axis=1)
        baselines={"checker":(p,out["threshold"]), "action_template_only":(ts,template_t),
                   "previous_half":(prev,.5), "previous_calibrated":(prev,prior_t),
                   "previous_and_source_veto":(((prev>=.5)&d["safe_mask"]).astype(float),.5),
                   "original_depth_veto":(d["safe_mask"].astype(float),.5),
                   "depth_point_intersection_only":((~body_seen).astype(float),.5)}
        case={"all":np.ones_like(y,bool),"translations":moving.any(1),
              "incomplete_depth_without_body_points":low,"observed_body_points":body_seen,
              "short_1_to_11":(a!=0).sum(1)<=11,"long_12_to_108":(a!=0).sum(1)>=12}
        counts={}
        for c,m in case.items():
            counts[c]={k:classification(y[m],s[m],t[m] if np.ndim(t) else t) for k,(s,t) in baselines.items()}
        first=d["first_blocked"]
        loc=(out["suspect_step"]>=0)&(first>0)
        counts["feedback"]=dict(decisions=dict(Counter(out["decision"])),blocked_with_first_step=int((first>0).sum()),
                                explicit_hints_on_true_collision=int(loc.sum()),
                                within_one_action=int((abs(first[loc]-1-out["suspect_step"][loc])<=1).sum()))
        counts["paired"]=bootstrap_diff(y,p,prev,d["house"])
        counts["row_count"]=len(y)
        counts["effective_houses"]=len(set(d["house"]))
        report["splits"][split]=counts
        for name in np.unique(d["file"]):
            ids=d["file"]==name
            payload={k:v[ids] for k,v in out.items()}
            payload.update(group=d["group"][ids],plan_id=d["plan_id"][ids],p_old=prev[ids])
            np.savez_compressed(PRIVATE/name,**payload)
    report["previous_matched_threshold"]=prior_t
    report["downstream_training_scores"]="Three-fold out-of-house predictions. Validation and test use the full training fit."
    (HERE/"checker_results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report["splits"]["test"]["all"]))

if __name__=="__main__":
    with threadpool_limits(limits=3):main()
