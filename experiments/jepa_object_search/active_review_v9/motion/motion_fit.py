"""Regularized candidate-centered comparison of four-step re-observation policies."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

MODES={"geometry":["geo"],"observed":["geo","observed"],"jepa":["geo","jepa"],"joint":["geo","observed","jepa"]}
RIDGE=(.03,.3,3.)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def dump(path,data):
    with path.open("x") as f:json.dump(data,f,indent=2,allow_nan=False)


def design(features, mode):
    x=np.concatenate([features[k] for k in MODES[mode]],-1).astype(float)
    return x-x.mean(0,keepdims=True)


def read(rows,root,split):
    for row in rows:
        key=digest(row["sid"])+".npz"
        with np.load(root/"features"/split/key) as z:f={k:z[k] for k in ("geo","observed","jepa","words","plans")}
        yield row["sid"],row["house"],f


def make_table(plan,root,split,subjects):
    by={r["sid"]:r for r in subjects}
    assert len(by)==len(plan) and set(by)=={r["sid"] for r in plan}
    table=[]
    for sid,home,f in read(plan,root,split):
        r=by[sid]
        y=np.asarray(r["gain"],float);valid=np.asarray(r["valid"],bool)
        assert y.shape==(6,16) and valid.shape==(16,) and y.min()>=0 and valid.any()
        table.append((sid,home,f,y,valid))
    return table


def fit(table,mode,alpha):
    x=[];yy=[];mass=[]
    counts=defaultdict(int)
    for _,h,_,_,_ in table:counts[h]+=1
    for _,h,f,y,valid in table:
        a=design(f,mode)[:,valid]
        b=(y-y.mean(0))[:,valid]
        x.append(a.reshape(-1,a.shape[-1]));yy.extend(b.flatten())
        mass.extend(np.full(b.size,1/(len(counts)*counts[h]*b.size)))
    x=np.concatenate(x);w=np.array(mass);yy=np.array(yy)
    scale=np.maximum(np.sqrt(w@(x*x)),.0001)
    z=x/scale
    coef=np.linalg.solve(z.T@(w[:,None]*z)+alpha*np.eye(x.shape[-1]),z.T@(w*yy))
    return dict(scale=scale.tolist(),coef=coef.tolist(),alpha=alpha)


def predicted(f,model,mode):
    return (design(f,mode)/np.asarray(model["scale"]))@np.asarray(model["coef"])


def policy_weights(scores):
    # Resolve numerical ties without favoring one of the six named routes.
    active=np.abs(scores-scores.max(0,keepdims=True))<=1e-8
    return active/active.sum(0,keepdims=True)


def house_mean(values):
    return np.mean([np.mean(v,axis=0) for v in values.values()],axis=0)


def fit_models(train,cal):
    fits={}
    for mode in MODES:
        cases=[]
        for alpha in RIDGE:
            model=fit(train,mode,alpha);by=defaultdict(list)
            for _,h,f,y,valid in cal:
                w=policy_weights(predicted(f,model,mode))
                by[h].append((w*y).sum(0)[valid].mean())
            cases.append((float(house_mean(by)),alpha,model))
        _,_,fits[mode]=max(cases,key=lambda c:(c[0],c[1]))
    by=defaultdict(list)
    for _,h,_,y,v in train:by[h].append(y[:,v].mean(1))
    action_means=house_mean(by)
    fits["common_route"]={"route":int(action_means.argmax()),"train_house_balanced":action_means.tolist()}
    return fits


def blind(plan,root,split,fits,dest):
    dest.mkdir(exist_ok=False)
    receipt={}
    for sid,h,f in read(plan,root,split):
        table={k:predicted(f,fits[k],k) for k in MODES};words=f["words"]
        table["training_common"]=np.zeros_like(table["joint"]);table["training_common"][fits["common_route"]["route"]]=1
        table["uniform"]=np.zeros_like(table["joint"])
        # A fixed no-goal displacement preference with unbiased ties.
        table["planned_displacement"]=(f["geo"][:,:,19])
        table["wrong_word"]=np.roll(table["joint"],1,axis=1)
        # With just one observed view the no-change prediction has identical text
        # features across candidates; within-state candidate centering removes it.
        table["joint_no_predicted_change"]=predicted({**f,"jepa":np.zeros_like(f["jepa"])},fits["joint"],"joint")
        file=dest/(digest(sid)+".npz");np.savez_compressed(file,words=words,**table)
        receipt[digest(sid)]=hashlib.sha256(file.read_bytes()).hexdigest()
    dump(dest/"manifest.json",receipt)


def evaluate(table,dest):
    scores=defaultdict(lambda:defaultdict(list));counts=defaultdict(int);class_std={}
    manifest=json.loads((dest/"manifest.json").read_text())
    for sid,h,f,y,v in table:
        file=dest/(digest(sid)+".npz");assert hashlib.sha256(file.read_bytes()).hexdigest()==manifest[digest(sid)]
        with np.load(file) as z:p={k:z[k] for k in z.files if k!="words"}
        p["oracle"]=y
        informative=(y.max(0)>y.min(0))&v
        counts["valid_start_queries"]+=int(v.sum());counts["all_zero_start_queries"]+=int(((y.max(0)==0)&v).sum())
        counts["informative_start_queries"]+=int(informative.sum());counts["informative_starts"]+=int(informative.any())
        counts["move_can_improve_over_both_turn_only"]+=int(((y[2:].max(0)>y[:2].max(0))&v).sum())
        counts["turn_only_can_improve_over_all_moving"]+=int(((y[:2].max(0)>y[2:].max(0))&v).sum())
        counts["same_camera_geometry_different_intermediate_views"]+=int(((y[4]!=y[5])&v).sum())
        for key,raw in p.items():
            weights=policy_weights(raw);gain=(weights*y).sum(0)
            anyhit=(weights*(y>0)).sum(0)
            for name,mask in [("all",v),("official",v&np.isin(f["words"],["chair.n.01","bed.n.01","sofa.n.01","toilet.n.02","television_receiver.n.01","potted_plant.n.01"])),
                              ("derived",v&~np.isin(f["words"],["chair.n.01","bed.n.01","sofa.n.01","toilet.n.02","television_receiver.n.01","potted_plant.n.01"]))]:
                if mask.any():scores[key,name][h].append([gain[mask].mean(),anyhit[mask].mean()])
    result=defaultdict(dict)
    for (key,name),houses in scores.items():
        a=house_mean(houses)
        result[name][key]=dict(mean_distinct=float(a[0]),mean_any=float(a[1]),house_distinct=[float(np.mean(houses[h],0)[0]) for h in sorted(houses)])
    return dict(counts=dict(counts),scores=dict(result))


def main(root,teacher):
    get=lambda name:json.loads((teacher/name).read_bytes())
    def unwrap(x):
        if isinstance(x,list):return x
        return x["rows"]
    plans={s:unwrap(get(s+".plan.json")) for s in ["train","calibration","holdout"]}
    houses=[{r["house"] for r in plan} for plan in plans.values()]
    assert list(map(len,houses))==[12,4,4] and len(set.union(*houses))==20
    data=lambda s:make_table(plans[s],root,s,unwrap(get(s+".json")))
    fitset=data("train");cal=data("calibration");models=fit_models(fitset,cal)
    dump(root/"coefficients.json",models)
    dest=root/"blind_test";blind(plans["holdout"],root,"holdout",models,dest)
    # The first access to held-out outcome labels happens after sealing every prediction.
    report=evaluate(data("holdout"),dest)
    report.update(model_sha=hashlib.sha256((root/"coefficients.json").read_bytes()).hexdigest(),
                  blind_sha=hashlib.sha256((dest/"manifest.json").read_bytes()).hexdigest(),
                  alpha={k:models[k]["alpha"] for k in MODES},train_common_route=models["common_route"]["route"])
    dump(root/"results.json",report)


if __name__=="__main__":main(Path(sys.argv[1]),Path(sys.argv[2]))
