"""Fit grounded opportunity heads on matched natural-HSSD action branches."""
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.nn import functional as F

from grounded_route import LocalGrounder,RouteVerifier
from features import build_features
from route_heads import load_heads
import metrics as stats
import argparse


MODES=("geometry","observed","jepa","joint")


def publish(path,record):
    if path.exists():raise FileExistsError(path)
    part=path.with_suffix(".temporary")
    part.write_text(json.dumps(record,indent=2,allow_nan=False,default=lambda x:x.item()))
    part.rename(path)


def pixel_loss(model,s):
    a,b=model.logits(s["code"],s["anchor_sim"],s["anchor_sim"][:,:,s["ids"]],s["text"])
    hit=(s["v"]/40).clamp(0,1)
    near=(s["n"]/s["v"].clamp_min(1)).clamp(0,1)
    visible=F.binary_cross_entropy_with_logits(a,hit,pos_weight=a.new_tensor(10.))
    weight=(s["v"]/40).clamp(0,1)
    near_loss=F.binary_cross_entropy_with_logits(b,near,reduction="none")
    return visible+.5*(weight*near_loss).sum()/weight.sum().clamp_min(1)


def grounding_accuracy(model,samples):
    count=torch.zeros(6,dtype=torch.float64)
    with torch.inference_mode():
        for s in samples:
            v,n=model(s["code"],s["anchor_sim"],s["anchor_sim"][:,:,s["ids"]],s["text"])
            for off,p,truth in ((0,v,s["v"]>=20),(3,n,s["n"]>=20)):
                active=p>=.5
                count[off:off+3]+=torch.tensor([(active&truth).sum(),(active&~truth).sum(),(~active&truth).sum()])
    out={}
    for pos,lab in [(0,"visible"),(3,"near")]:
        tp,fp,fn=count[pos:pos+3].tolist()
        out[lab]=dict(tp=int(tp),fp=int(fp),fn=int(fn),f1=2*tp/max(1,2*tp+fp+fn))
    return out


def loss(net,g,k):
    dense,j,geom,_,_=g["modes"][k]
    logits,near,pr=net(dense,j,geom)
    y=g["y"];i,l=torch.triu(g["mask"],diagonal=1).nonzero(as_tuple=True)
    d=y[i]-y[l];ds=logits[i]-logits[l];different=d.ne(0)
    def avg(a,m):
        return (a*m).sum()/m.sum().clamp_min(1)
    target=(y>0).float()
    bc=F.binary_cross_entropy_with_logits(logits,target,pos_weight=logits.new_tensor(7.))
    rank=avg(F.softplus(-ds*d.sign()),different)
    eq=g["matched"][i,l]
    matched=avg(F.softplus(-ds*d.sign()),different&eq[:,None])
    pair=(d[:,:,None]>0)&(d[:,None,:]<0)&eq[:,None,None]
    inter=avg(F.softplus(-(ds[:,:,None]-ds[:,None,:])),pair)
    loss=.4*bc+.6*rank+.6*matched+.75*inter
    if g["family"]=="primitives":
        loss=loss+.4*F.binary_cross_entropy_with_logits(near,g["near"],pos_weight=near.new_tensor(5.))
    return loss+.25*avg(F.binary_cross_entropy_with_logits(pr,g["progress"],reduction="none"),g["forward"])


def permutation(query,domain,sid,h):
    perm=dict()
    for d in ("official","derived"):
        order=[i for i,v in enumerate(domain) if v==d]
        random.Random(f"83|{sid}|{h}|{d}").shuffle(order)
        perm.update(dict(zip(order,order[1:]+order[:1])))
    return [perm[i] for i in range(len(query))]


@torch.inference_mode()
def evaluate(networks,groups,all_arms=True):
    raw=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(list))))
    rank=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(list))))
    for g in groups:
        perm=permutation(g["query"],g["domain"],g["sid"],g["h"])
        for mi,(dense,j,geom,nearest,v6) in enumerate(g["modes"]):
            arms={k:net(dense,j,geom)[0].sigmoid() for k,net in networks.items()}
            arms["wrong"]=arms["joint"][:,perm]
            arms["displacement"]=geom[:,-1,5,None].expand_as(arms["joint"])
            if all_arms:
                arms["v6"]=v6
                arms["joint_no_future"]=networks["joint"](dense,nearest,geom)[0].sigmoid()
                nd,nj=g["null_modes"][mi]
                arms["no_word"]=networks["joint"](nd,nj,geom)[0].sigmoid().expand_as(v6)
                arms["retain"]=torch.zeros_like(v6)
            for dom in ("official","derived"):
                idx=[i for i,v in enumerate(g["domain"]) if v==dom]
                if not idx:continue
                if len(idx)<2:raise ValueError("cannot compare within-domain wrong goal")
                key=g["family"]+"|"+dom
                yy=g["y"][:,idx]; ii,ll=torch.triu(g["matched"],diagonal=1).nonzero(as_tuple=True)
                d=yy[ii]-yy[ll]
                opposite=(d[:,:,None]>0)&(d[:,None,:]<0)
                for arm,s in arms.items():
                    sc=s[:,idx]
                    z=stats.per_origin(sc,g,yy)
                    z["already_eligible_at_start_rate"]=0. # unused by the V8 outcome
                    raw[key][arm][g["house"]][g["sid"]].append(z)
                    sd=sc[ii]-sc[ll]
                    mask=d.ne(0)
                    if mask.any():
                        product=sd[mask]*d[mask].sign()
                        rate=float(((product>1e-7)+.5*(product.abs()<=1e-7)).mean())
                        dd=(sd[:,:,None]-sd[:,None,:])[opposite]
                        dd_rate=float(((dd>1e-7)+.5*(dd.abs()<=1e-7)).mean()) if dd.numel() else None
                        rank[key][arm][g["house"]][g["sid"]].append((rate,dd_rate))
    result={}
    for key,arms in raw.items():
        out=stats.aggregate(arms)
        for arm,info in out.items():
            info["mean"].pop("already_eligible_at_start_rate",None)
            for house in info["houses"].values():house.pop("already_eligible_at_start_rate",None)
            evidence=rank[key][arm]; vals=[]; opos=[]; total=0; total_opp=0
            for starts in evidence.values():
                rates=[]; d_rates=[]
                for records in starts.values():
                    rates.append(np.mean([a for a,_ in records]));total+=1
                    dd=[b for _,b in records if b is not None]
                    if dd:d_rates.append(np.mean(dd));total_opp+=1
                if rates:vals.append(np.mean(rates))
                if d_rates:opos.append(np.mean(d_rates))
            info["matched_nominal_geometry"]=dict(rank=float(np.mean(vals)) if vals else None,
                  double_difference=float(np.mean(opos)) if opos else None,ranking_starts=total,opposite_starts=total_opp)
        result[key]=out
    return result


def select(result):
    vals=[];cnt=[]
    for domain in ("official","derived"):
        g=result["primitives|"+domain]
        m=lambda a:g[a]["mean"]["gain"]
        vals.append(m("joint")-max(m(a) for a in ("wrong","displacement","observed")))
        cnt.append(m("joint"))
    return min(vals),sum(cnt)/2


def load_partition(path):
    bundle = torch.load(path, weights_only=True, map_location="cpu")
    samples = bundle["states"]
    if not samples:
        raise ValueError("empty partition")
    return bundle["anchors"], bundle["queries"], samples


def fit(args):
    anc,names,train=load_partition(args.train)
    cal_anchors,cal_names,cal=load_partition(args.calibration)
    if cal_names!=names or not torch.equal(cal_anchors,anc): raise ValueError("inconsistent ordered anchor bank")
    if any(s["family"]!="primitives" for s in cal): raise ValueError("calibration selection uses fixed motion primitives")
    D=args.output
    D.mkdir(parents=True,exist_ok=True)
    a={s["house"] for s in train};b={s["house"] for s in cal}
    assert len(a)==12 and len(b)==4 and not a&b
    grounding=LocalGrounder(anc)
    opt=torch.optim.AdamW(grounding.parameters(),lr=.002,weight_decay=.015)
    subset=[s for s in train if s["family"]=="primitives"]
    rng=random.Random(83)
    for epoch in range(12):
        rng.shuffle(subset)
        for start in range(0,len(subset),6):
            term=torch.stack([pixel_loss(grounding,s) for s in subset[start:start+6]]).mean()
            opt.zero_grad(set_to_none=True);term.backward()
            torch.nn.utils.clip_grad_norm_(grounding.parameters(),3,error_if_nonfinite=True);opt.step()
    grounding.eval()
    gr=grounding_accuracy(grounding,cal)
    print("grounding",gr,flush=True)
    bases=load_heads(args.full,args.sparse)
    train_groups=[g for s in train for g in build_features(bases,grounding,s,anc)]
    val_groups=[g for s in cal for g in build_features(bases,grounding,s,anc)]
    by_family={f:[g for g in train_groups if g["family"]==f] for f in ("frontiers","primitives")}
    print("fit",[(k,len(v)) for k,v in by_family.items()],flush=True)
    nets={name:RouteVerifier(name) for name in MODES}
    for name in MODES[1:]:nets[name].load_state_dict(nets[MODES[0]].state_dict())
    optim={k:torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.015) for k,net in nets.items()}
    best=None;history=[]
    for ep in range(1,13):
        # Equal family mass. Every original and primitive group is seen at least once per epoch.
        count=max(len(v) for v in by_family.values());pool=[]
        for v in by_family.values():
            pool.extend(v);pool.extend(rng.choices(v,k=count-len(v)))
        rng.shuffle(pool)
        for z in range(0,len(pool),16):
            batch=[(g,rng.randrange(4)) for g in pool[z:z+16]]
            for name,net in nets.items():
                term=torch.stack([loss(net,g,k) for g,k in batch]).mean()
                optim[name].zero_grad(set_to_none=True);term.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),3,error_if_nonfinite=True);optim[name].step()
        if ep%4==0:
            values=evaluate(nets,val_groups,False);key=(*select(values),-ep)
            history.append(dict(epoch=ep,criterion=key[:2],results={
                d:{a:x["mean"]["gain"] for a,x in g.items()} for d,g in values.items()}))
            if best is None or key>best[0]:
                best=key,ep,{k:copy.deepcopy(net.state_dict()) for k,net in nets.items()}
            print("cal",ep,key,flush=True)
    _,ep,selected=best
    for k,net in nets.items():net.load_state_dict(selected[k])
    ck=D/"v8.pt"
    if ck.exists():raise FileExistsError(ck)
    torch.save(dict(grounding=grounding.state_dict(),heads=selected,queries=names),ck)
    directory=Path(__file__).resolve().parent
    src={k:hashlib.sha256((directory/k).read_bytes()).hexdigest() for k in
         ("grounded_route.py","features.py","dense_clip.py","train_v8.py","protocol.json")}
    publish(D/"selection.json",dict(epoch=ep,weights_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),
            source_sha256=src,train_houses=sorted(a),calibration_houses=sorted(b),grounding=gr,history=history,
            calibration=evaluate(nets,val_groups),training_groups={k:len(v) for k,v in by_family.items()}))
    print("frozen epoch",ep,flush=True)


def restore_heads(path):
    ck=torch.load(path,weights_only=True,map_location="cpu")
    ground=LocalGrounder(ck["grounding"]["anchors"]);ground.load_state_dict(ck["grounding"]);ground.eval()
    networks={k:RouteVerifier(k) for k in ck["heads"]}
    for k,v in networks.items():v.load_state_dict(ck["heads"][k]);v.eval()
    return ground,networks,ck["queries"]


def check(args):
    D=args.output
    ck=D/"v8.pt";selection=json.loads((D/"selection.json").read_text())
    assert hashlib.sha256(ck.read_bytes()).hexdigest()==selection["weights_sha256"]
    ground,nets,names=restore_heads(ck)
    anchors,labels,test=load_partition(args.holdout)
    if labels!=names or not torch.allclose(ground.anchors,anchors,atol=1e-6): raise ValueError("invalid anchor bank")
    if any(s["family"]!="primitives" for s in test): raise ValueError("confirmatory check uses fixed motion primitives")
    houses=sorted({s["house"] for s in test})
    assert len(houses)==4 and not set(houses)&set(selection["train_houses"]+selection["calibration_houses"])
    base=load_heads(args.full,args.sparse)
    groups=[g for s in test for g in build_features(base,ground,s,anchors)]
    score=evaluate(nets,groups)
    domains=["primitives|official","primitives|derived"]
    gain=lambda d,a:score[d][a]["mean"]["gain"]
    joint_plus=[d for d in domains if gain(d,"joint")>max(gain(d,"observed"),gain(d,"jepa"))+1e-10 and
                sum(score[d]["joint"]["houses"][h]["gain"]>score[d]["observed"]["houses"][h]["gain"]+1e-10 for h in houses)>=2]
    ranks=[score[d]["joint"]["matched_nominal_geometry"]["rank"] for d in domains]
    gates=dict(official_positive=gain(domains[0],"joint")>0,increment_and_houses=bool(joint_plus),
               goal_both_domains=all(gain(d,"joint")>gain(d,"wrong")+1e-10 for d in domains),
               beats_geometry=any(gain(d,"joint")>max(gain(d,"displacement"),gain(d,"geometry"))+1e-10 for d in domains),
               matched_pair_rank=any(v is not None and v>.5+1e-10 for v in ranks))
    publish(D/"independent.json",dict(passed=all(gates.values()),checks=gates,metrics=score,
            starts=len(test),houses=houses,grounding=grounding_accuracy(ground,test),ck_sha256=selection["weights_sha256"]))
    print(gates,flush=True)


if __name__=="__main__":
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(83)
    torch.use_deterministic_algorithms(True)
    ap=argparse.ArgumentParser(description=__doc__)
    subs=ap.add_subparsers(dest="mode",required=True)
    for name in ("fit","check"):
        sp=subs.add_parser(name)
        for key in ("full","sparse","output"):
            sp.add_argument("--"+key,type=Path,required=True)
        for key in (("train","calibration") if name=="fit" else ("holdout",)):
            sp.add_argument("--"+key,type=Path,required=True)
    args=ap.parse_args()
    if args.mode=="fit": fit(args)
    else: check(args)
