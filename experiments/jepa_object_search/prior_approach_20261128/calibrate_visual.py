"""Calibrate near-context log-odds using actual HSSD branch outcomes and RGB encodings."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from placement_prior import PlacementPrior


def split_data(project, model, prior, split):
    cache=project/("jepa_hidden_v11" if split=="train" else "jepa_route_v12")/"models/cache/hssd"/split
    rows={arm:[] for arm in ("jepa","direct")}
    y=[]; houses=[]; targets=[]
    for path in sorted((cache/"actor").glob("*.npz")):
        with np.load(path) as a:
            ctx,plans,words,house=a["ctx"],a["actions"],a["words"].tolist(),str(a["house"])
            text=a["text"][0]
        if not words:
            continue
        with np.load(cache/"truth"/path.name) as a:
            found=(a["y"].T>0).astype(float)
        current=prior.current_cues(ctx,prior.visual_prompts(model))
        cues=prior.local_directions(plans,current)
        out=model.cached(ctx,text,plans,words[0])
        # Future room logits depend only on the encoded departure/plan, not target.
        for i,word in enumerate(words):
            if prior.key(word) is None:
                continue
            for arm in rows:
                rows[arm].append(prior.features(word,out["env_pred_"+arm],cues))
            y.append(found[i]);houses.append(house);targets.append(word)
    if not y:
        raise ValueError("missing cached trajectories for "+split)
    return {k:np.stack(v) for k,v in rows.items()},np.stack(y),np.array(houses),np.array(targets)


def fit(x,y,h):
    x=np.concatenate([np.ones_like(x[...,:1]),x],-1)
    # Equal house weight. Centering gives a within-query branch preference term
    # as well as the common event scale. Same objective for both arms.
    weights=np.array([1/(h==v).sum() for v in h]);weights/=weights.sum()
    pair=(y[:,:,None]>y[:,None,:])
    dx=x[:,:,None]-x[:,None,:]
    def loss(w):
        z=x@w
        cls=((np.logaddexp(0.,z)-z*y).mean(1)*weights).sum()
        rank=(np.sum(np.logaddexp(0.,-(dx@w))*pair,(1,2))*weights).sum()/max(1.,(pair.sum((1,2))*weights).sum())
        return cls+rank+0.002*(w[1:]**2).sum()
    opt=minimize(loss,[0.,0.25,0.25],method="L-BFGS-B",bounds=[(-8,8),(0,5),(0,8)])
    if not opt.success:
        raise RuntimeError(opt.message)
    return opt.x


def assess(x,y,h,w):
    output={}
    for tag,d in (("with_structure",w[2]),("without_structure",0.)):
        z=w[0]+w[1]*x[:,:,0]+d*x[:,:,1]
        prob=expit(z)
        picks=z.argmax(1)
        valid=y[np.arange(len(y)),picks]
        output[tag]=dict(hits=int(valid.sum()),queries=len(y),house_macro=float(np.mean([valid[h==v].mean() for v in set(h)])),
                         brier=float(((prob-y)**2).mean()))
    return output


def main(project, out):
    sys.path.insert(0,str(project/"jepa_env_v13/models"))
    from env_routes import EnvironmentRoutes
    torch.set_num_threads(2)
    model=EnvironmentRoutes(project,11)
    pf=out/"learned_prior.json"
    prior=PlacementPrior(json.loads(pf.read_text()))
    parts=[split_data(project,model,prior,s) for s in ("train","extra_train")]
    y=np.concatenate([p[1] for p in parts]);h=np.concatenate([p[2] for p in parts])
    assert len(set(h))==40
    arms={}
    for arm in ("jepa","direct"):
        x=np.concatenate([p[0][arm] for p in parts])
        arms[arm]=dict(coefficients=fit(x,y,h).tolist())
    record=dict(prior_sha256=hashlib.sha256(pf.read_bytes()).hexdigest(),arms=arms,train_houses=sorted(set(h)),
                train_queries=len(y),event="at least one surveyed observation of this target within 2m and >=32 target pixels along a 24-step candidate; fitted on hidden branch contrast tasks")
    path=out/"context_calibration.json"
    data=json.dumps(record,indent=2)
    if path.exists() and path.read_text()!=data:
        raise ValueError("different frozen weights")
    path.write_text(data)
    test,truth,held,t=split_data(project,model,prior,"fresh2")
    assert not set(held)&set(h)
    summary={"queries":len(t),"houses":len(set(held)),"overlap":0,"versions":{}}
    for arm in arms:
        summary["versions"][arm]=assess(test[arm],truth,held,arms[arm]["coefficients"])
    (out/"visual_validation.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({"arms":arms,"summary":summary}),flush=True)


if __name__=="__main__":
    main(Path(sys.argv[1]),Path(sys.argv[2]))
