"""Predeclared combined fitting on local detours and natural/perturbed long trajectories."""
import hashlib,json,os
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from checker import PathChecker,check_inputs,summarize,per_step
from mixed_checker import MixedPathChecker
from path_geometry import NAMES
from train_checker import cutoff,classification,template_features,bootstrap_diff

HERE=Path(__file__).resolve().parent
SEED=1133
OUT=HERE/"private_checker_mixed"


def read(name,kind):
    folder=HERE/("private_natural" if kind=="natural" else "private_data")
    files=sorted(p for p in folder.glob(name+"_*.npz") if ".partial." not in p.name)
    assert len(files)=={"train":12,"val":6,"test":12,"blind":6}[name],(name,kind,len(files))
    blocks=[]
    for p in files:
        with np.load(p,allow_pickle=False) as z:
            b={k:z[k] for k in ("action","path","first_blocked","safe_mask","reach","house")}
            b["kind"]=z["kind"].astype(str) if kind=="natural" else np.repeat("local",len(b["reach"]))
            check_inputs(b["action"],b["path"])
            assert np.all(z["split"]==name) and len(np.unique(b["house"]))==1
            blocks.append(b)
    return combine(blocks)


def combine(blocks):
    d={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    d["reach"]=d["reach"].astype(bool)
    return d


def weights(d):
    # Equal total training weights for local detours and natural-collection rows.
    # No data-source tag is passed to a learned model or used for deployment thresholds.
    local=d["kind"]=="local"
    return np.where(local,.5/max(.001,local.mean()),.5/max(.001,1-local.mean()))


def tree(max_leaf_nodes,iterations):
    return HistGradientBoostingClassifier(max_leaf_nodes=max_leaf_nodes,max_iter=iterations,
            l2_regularization=2,early_stopping=False,random_state=SEED)


def select_policy(y,p):
    threshold=cutoff(y,p)
    order=np.argsort(-p,kind="stable")
    s=p[order];end=np.flatnonzero(np.r_[np.diff(s)!=0,True])
    n=end+1
    positive=np.cumsum(y[order])[end]
    valid=end[(n>=30)&(positive/n>=.85)]
    upper=None if not len(valid) else float(s[valid[-1]])
    order=np.argsort(p,kind="stable")
    s=p[order];end=np.flatnonzero(np.r_[np.diff(s)!=0,True]);n=end+1
    neg=np.cumsum(~y[order])[end]
    eligible=(n>=30)&(neg/n>=.90)
    if upper is not None:eligible&=s[end]<upper
    valid=end[eligible]
    lower=None if not len(valid) else float(s[valid[-1]])
    return dict(advisory=threshold,admit=upper,block=lower,validation_n=len(y))


def select_report(y,status):
    yes=status=="admit"; no=status=="block"; unknown=status=="uncertain"
    return dict(n=len(y),admitted=int(yes.sum()),blocked=int(no.sum()),deferred=int(unknown.sum()),
                admit_correct=int(y[yes].sum()),block_correct=int((~y[no]).sum()),
                admit_precision=float(y[yes].mean()) if yes.any() else None,
                block_precision=float((~y[no]).mean()) if no.any() else None,
                deferred_feasible=int(y[unknown].sum()),coverage=float((yes|no).mean()))


def eval_set(d,checker,v1,action_model,template_t):
    a,x,y=d["action"],d["path"],d["reach"]
    second=checker.predict(a,x)
    first=v1.predict(a,x)
    tp=action_model.predict_proba(template_features(a,x))[:,1]
    observed=((x[:,:,13]>.5)&(a==1)).any(1)
    n=(a!=0).sum(1)
    tt=np.where(n<=36,template_t["near"],template_t["long"])
    models=dict(mixed=(second["p_reach"],second["threshold"]),local_fit_only=(first["p_reach"],first["threshold"]),
                template=(tp,tt),point_only=((~observed).astype(float),.5),
                coverage_threshold=(d["safe_mask"].astype(float),.5))
    groups={"all":np.ones(len(y),bool),"12":n==12,"24":n==24,"52":n==52,"80":n==80,"108":n==108}
    for k in np.unique(d["kind"]):groups["source_"+k]=d["kind"]==k
    groups["no_body_point"]=~observed
    groups["observed_body_point"]=observed
    sections={}
    for name,m in groups.items():
        if not m.any():continue
        sections[name]={k:classification(y[m],p[m],t[m] if np.ndim(t) else t) for k,(p,t) in models.items()}
        sections[name]["selective"]=select_report(y[m],second["screening"][m])
    return dict(conditional=sections,paired_vs_v1=bootstrap_diff(y,second["p_reach"],first["p_reach"],d["house"]))


def fam(d):return {h.split("_")[0] for h in d["house"]}


def main():
    os.umask(0o077)
    OUT.mkdir(exist_ok=True)
    tr=combine([read("train","local"),read("train","natural")])
    va=combine([read("val","local"),read("val","natural")])
    assert not fam(tr)&fam(va)
    a,x=tr["action"],tr["path"]
    w=weights(tr)
    route=tree(15,110).fit(summarize(a,x),tr["reach"],sample_weight=w)
    control=tree(15,110).fit(template_features(a,x),tr["reach"],sample_weight=w)
    step=np.arange(a.shape[1])[None,:]+1
    first=tr["first_blocked"][:,None]
    valid=(a==1)&((first==0)|(step<=first))
    local=tree(11,80).fit(per_step(a,x)[valid],(step==first)[valid],sample_weight=np.broadcast_to(w[:,None],a.shape)[valid])
    vscore=route.predict_proba(summarize(va["action"],va["path"]))[:,1]
    vscore[~(va["action"]==1).any(1)]=1.
    near=(va["action"]>0).sum(1)<=36
    policy={k:select_policy(va["reach"][m],vscore[m]) for k,m in (("near",near),("long",~near))}
    tp=control.predict_proba(template_features(va["action"],va["path"]))[:,1]
    template_t={k:cutoff(va["reach"][m],tp[m]) for k,m in (("near",near),("long",~near))}
    model_file=OUT/"checker.joblib"
    joblib.dump(dict(names=NAMES,route=route,local=local,threshold=.5,operating_points=policy),model_file)
    fingerprint=hashlib.sha256(model_file.read_bytes()).hexdigest()
    checker=MixedPathChecker.load(model_file)
    v1=PathChecker.load(HERE/"private_checker/checker.joblib")
    report=dict(seed=SEED,one_fit=True,feature_names=NAMES,frozen_before_test_sha256=fingerprint,
                train_houses=len(fam(tr)),calibration_houses=len(fam(va)),
                train_rows=len(tr["reach"]),calibration_rows=len(va["reach"]),
                length_cutoff=36,policy=policy,template_threshold=template_t,
                selection="Advisory cutoff maximizes true-pass acceptance given <=15% false permit in calibration duration stratum. Selective admission requires >=85% precision for >=30 proposals, denial >=90% for >=30; all others deferred. Near <=36 actions, long >36. No split, family or generation label reaches inference.",
                conditions={})
    # Models and thresholds are irreversible for this run before evaluating either held-out set.
    oldtest=None
    for name,split,source in (("prior_12_house_local","test","local"),("prior_12_house_natural","test","natural"),("new_6_house_blind","blind","natural")):
        d=read(split,source)
        assert not (fam(tr)|fam(va))&fam(d)
        if split=="blind":assert not oldtest&fam(d)
        else:oldtest=fam(d)
        report["conditions"][name]=eval_set(d,checker,v1,control,template_t)
    assert hashlib.sha256(model_file.read_bytes()).hexdigest()==fingerprint
    (HERE/"mixed_results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for k,v in report["conditions"].items():
        print(k,v["conditional"]["all"]["mixed"],v["conditional"]["all"]["selective"],flush=True)

if __name__=="__main__":
    with threadpool_limits(limits=3):main()
