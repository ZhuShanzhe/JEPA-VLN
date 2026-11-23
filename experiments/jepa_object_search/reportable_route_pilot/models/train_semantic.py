"""Fit and independently evaluate an action-by-goal semantic fork predictor."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.nn import functional as F

from semantic_head import SemanticFork,compose,displacement,restore


SOURCE = Path(__file__).resolve().parent
D = SOURCE
V7 = None
import train_report_head as data
from report_head import load_v7,route_scores
from route_heads import route_logits


def pairs(g):
    i,j=torch.triu(g["mask"],diagonal=1).nonzero(as_tuple=True)
    return i,j


def loss(net,g,mode):
    p,near,motion=g["modes"][mode]
    semantic,pred=net(p,near,motion,g["text"])
    y=g["y"].clamp_max(2).sqrt()
    dc=lambda x:x-x.mean(0,keepdim=True)-x.mean(1,keepdim=True)+x.mean()
    target=dc(y)
    weight=1+4*(target.abs()>.10)
    regression=(weight*F.smooth_l1_loss(dc(semantic),2*target,reduction="none")).sum()/weight.sum()
    i,j=pairs(g)
    dy=y[i]-y[j]; ds=semantic[i]-semantic[j]
    active=dy.ne(0)
    def avg(value,sel):
        return (value*sel).sum()/sel.sum().clamp_min(1)
    ranking=avg(F.softplus(-ds*dy.sign()),active)
    opposite=(dy[:,:,None]>0)&(dy[:,None,:]<0)
    double=ds[:,:,None]-ds[:,None,:]
    word_pair=avg(F.softplus(-double),opposite)
    e=pred-g["truth"]
    return (.8*regression+.50*ranking+1.25*word_pair+
            .25*(1-(pred*g["truth"]).sum(-1)).mean()+.15*(e-e.mean(0,keepdim=True)).square().sum(-1).mean())


def is_opposite(g):
    i,j=pairs(g)
    d=g["y"][i]-g["y"][j]
    return bool(((d>0).any(1)&(d<0).any(1)).any())


def conflict_score(p,g,idx):
    i,j=pairs(g)
    y=g["y"][:,idx]
    dp=p[i]-p[j];dy=y[i]-y[j]
    x,q,r=((dy[:,:,None]>0)&(dy[:,None,:]<0)).nonzero(as_tuple=True)
    if not len(x):
        return None
    first,second=dp[x,q],dp[x,r]
    score=first-second
    return dict(dd=float(((score>1e-7)+.5*(score.abs()<=1e-7)).mean()),
                both=float(((first>1e-7)&(second< -1e-7)).float().mean()),
                count=len(x))


def collect_conflict(lines):
    households=[]
    starts=0;pair_masks=0
    for house,by_start in sorted(lines.items()):
        start_means=[]
        for sid,rows in by_start.items():
            if rows:
                start_means.append([np.mean([r[k] for r in rows]) for k in ("dd","both")])
                pair_masks+=sum(r["count"] for r in rows)
        if start_means:
            starts+=len(start_means);households.append(np.mean(start_means,axis=0))
    vals=np.mean(households,axis=0) if households else (None,None)
    return dict(double_difference=vals[0],both_directions=vals[1],opposite_starts=starts,
                opposite_houses=len(households),opposite_query_pairs_including_view_masks=pair_masks)


@torch.inference_mode()
def evaluate(net,groups,lam,v7,v7alpha,want_all):
    raw=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(list))))
    conflict=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(list))))
    for g in groups:
        for forecast,near,motion in g["modes"]:
            e,_=net(forecast,near,motion,g["text"])
            p,zero=compose(e,motion,lam)
            arms=dict(correct=p,wrong=p[:,g["wrong"]],no_word=zero,
                      displacement=displacement(motion)[:,None].expand_as(p))
            if want_all:
                obs,_=net(near,near,motion,g["text"],observed_only=True)
                arms.update(observed=compose(obs,motion,lam)[0],v6=route_logits(forecast,g["text"]).sigmoid(),
                            v7=route_scores(v7,forecast,near,motion,g["text"],v7alpha),retain=torch.zeros_like(p))
            for domain in ("official","derived","joint"):
                idx=[i for i,x in enumerate(g["domain"]) if domain=="joint" or domain==x]
                for arm,s in arms.items():
                    piece=s[:,idx]
                    if domain!="joint":
                        entry=data.per_origin(piece,g,g["y"][:,idx])
                        entry["already_eligible_at_start_rate"]=float((g["initial"][idx]>0).float().mean())
                        raw[domain][arm][g["house"]][g["sid"]].append(entry)
                    c=conflict_score(piece,g,idx)
                    if c:
                        conflict[domain][arm][g["house"]][g["sid"]].append(c)
    out={key:data.aggregate(table) for key,table in raw.items()}
    out["conflict"]={domain:{arm:collect_conflict(value) for arm,value in methods.items()}
                     for domain,methods in conflict.items()}
    if want_all:
        for domain,methods in raw.items():
            for arm in methods:
                if arm=="correct":continue
                diffs=[]
                for hid,starts in methods["correct"].items():
                    for sid,rows in starts.items():
                        other=methods[arm][hid][sid]
                        diffs.append(np.mean([a["instances"]-b["instances"] for a,b in zip(rows,other)]))
                out[domain]["correct"].setdefault("paired_starts",{})[arm]=dict(
                    win=sum(bool(x>1e-8) for x in diffs),tie=sum(bool(abs(x)<=1e-8) for x in diffs),
                    loss=sum(bool(x < -1e-8) for x in diffs))
    return out


def gain(result,domain,arm):
    return result[domain][arm]["mean"]["gain"]


def selector(result):
    gaps=[]
    for dom in ("official","derived"):
        gaps.append(gain(result,dom,"correct")-max(gain(result,dom,b) for b in ("wrong","no_word","displacement")))
    dd=result["conflict"]["joint"]["correct"]["double_difference"]
    return min(gaps)+.02*(dd-.5)


def fit(heads,v7,v7alpha):
    train,calib=[data.prepare(s,heads) for s in ("train","calibration")]
    train_h=sorted({g["house"] for g in train})
    calib_h=sorted({g["house"] for g in calib})
    if set(train_h)&set(calib_h):
        raise ValueError("overlapping training and calibration scenes")
    distinct={q:t for g in train for q,t in zip(g["query"],g["text"])}
    words=sorted(distinct)
    net=SemanticFork(heads[0].center,heads[0].basis,torch.stack([distinct[x] for x in words]))
    net.residual.load_state_dict(v7.residual.state_dict())
    opp=[g for g in train if is_opposite(g)]
    table=dict(groups=len(train),opposite_groups=len(opp),opposite_physical_starts=len({g["sid"] for g in opp}),
               train_houses=train_h,calibration_houses=calib_h,anchors=words)
    print("train",table,flush=True)
    if not opp:
        raise ValueError("missing conflicting-goal supervision")
    rng=random.Random(47)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.02)
    best=None;rec=[]
    for ep in range(1,13):
        net.train();rng.shuffle(train)
        for j in range(0,len(train),12):
            batch=train[j:j+12]
            extra=rng.choices(opp,k=len(batch))
            value=torch.stack([loss(net,g,rng.randrange(4)) for g in batch+extra]).mean()
            opt.zero_grad(set_to_none=True);value.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),3,error_if_nonfinite=True);opt.step()
        if ep%4==0:
            for lam in (0.,.75,1.5):
                metrics=evaluate(net.eval(),calib,lam,v7,v7alpha,False)
                val=selector(metrics)
                key=(val,gain(metrics,"official","correct"),-lam,-ep)
                if best is None or key>best[0]:
                    best=(key,copy.deepcopy(net.state_dict()),lam,ep)
                rec.append(dict(epoch=ep,strength=lam,value=val,
                                official={k:v["mean"]["gain"] for k,v in metrics["official"].items()},
                                derived={k:v["mean"]["gain"] for k,v in metrics["derived"].items()},
                                joint_conflict=metrics["conflict"]["joint"]["correct"]))
            print("epoch",ep,"selector",[round(x["value"],5) for x in rec[-3:]],flush=True)
    _,weights,lam,ep=best
    if (D/"semantic.pt").exists():raise FileExistsError(D/"semantic.pt")
    torch.save(dict(model=weights,strength=lam),D/"semantic.pt");net.load_state_dict(weights)
    out=dict(data=table,epoch=ep,strength=lam,calibration_choices=rec,
             calibration=evaluate(net.eval(),calib,lam,v7,v7alpha,True),
             protocol_sha256=data.sha(SOURCE/"v7s_protocol.json"),sha256=data.sha(D/"semantic.pt"),
             fit_inputs={f"{p.name}/{s}":data.sha(p/f"{s}.rows.json") for p in (data.DATA,data.AUX)
                         for s in ("train","calibration")},
             source_hashes={f:data.sha(SOURCE/f) for f in ("train_semantic.py","semantic_head.py")})
    data.write_new(D/"selected.json",out)
    print("selected",ep,lam,out["sha256"],flush=True)


def check(heads,v7,v7alpha,official,derived):
    sel=json.loads((D/"selected.json").read_text())
    if sel["sha256"]!=data.sha(D/"semantic.pt") or sel["protocol_sha256"]!=data.sha(SOURCE/"v7s_protocol.json"):
        raise ValueError("frozen artifact has changed")
    data.DATA,data.AUX=official,derived
    groups=data.prepare("holdout",heads)
    houses={g["house"] for g in groups}
    if houses&set(sel["data"]["train_houses"]+sel["data"]["calibration_houses"]):
        raise ValueError("split overlap")
    orig=set(json.loads((V7/"result.json").read_text())["score"]["v7"]["houses"])
    if set(orig)&houses or len(houses)!=4:
        raise ValueError("second independent house panel required")
    net,lam=restore(D/"semantic.pt")
    scores=evaluate(net,groups,lam,v7,v7alpha,True)
    j=scores["conflict"].get("joint",{}); main=j.get("correct",{}); obs=j.get("observed",{})
    c=main.get("double_difference");b=obs.get("double_difference")
    go=lambda dom,k:gain(scores,dom,"correct")>gain(scores,dom,k)+1e-10
    gates=dict(opportunity=main.get("opposite_houses",0)>=2,
               official_positive=gain(scores,"official","correct")>0,
               official_vs_wrong=go("official","wrong"),official_vs_no_word=go("official","no_word"),
               official_vs_v6=go("official","v6"),derived_vs_wrong=go("derived","wrong"),
               derived_vs_no_word=go("derived","no_word"),
               goal_interaction=c is not None and b is not None and c>max(.5,b)+1e-10,
               displacement=any(go(d,"displacement") for d in ("official","derived")))
    gates={key:bool(value) for key,value in gates.items()}
    data.write_new(D/"independent.json",dict(pass_gate=all(gates.values()),checks=gates,
                   score=scores,starts=len({g["sid"] for g in groups}),houses=sorted(houses),
                   selected_sha256=data.sha(D/"selected.json"),labels={p.name:data.sha(p/"holdout.rows.json")
                                                                      for p in (official,derived)}))
    print("independent",gates,flush=True)


if __name__=="__main__":
    parse=argparse.ArgumentParser(description=__doc__)
    parse.add_argument("stage",choices=("fit","check"))
    parse.add_argument("--full-checkpoint",type=Path,required=True)
    parse.add_argument("--sparse-checkpoint",type=Path,required=True)
    parse.add_argument("--feature-dir",type=Path,required=True)
    parse.add_argument("--output",type=Path,required=True)
    parse.add_argument("--v7-run",type=Path,required=True)
    parse.add_argument("--official",type=Path,required=True)
    parse.add_argument("--derived",type=Path,required=True)
    args=parse.parse_args()
    D=args.output.resolve()
    D.mkdir(parents=True,exist_ok=True)
    V7=args.v7_run.resolve()
    data.DATA=args.official.resolve()
    data.AUX=args.derived.resolve()
    data.FEATURES=args.feature_dir.resolve()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(47)
    torch.use_deterministic_algorithms(True)
    data.HERE=D
    heads=data.load_heads(args.full_checkpoint,args.sparse_checkpoint)
    v7,v7alpha=load_v7(V7/"v7.pt")
    if args.stage=="fit":
        fit(heads,v7,v7alpha)
    else:
        check(heads,v7,v7alpha,args.official,args.derived)
