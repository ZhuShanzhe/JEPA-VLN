"""Optional frozen synthetic-layout intervention and ten-epoch replay adaptation."""
from collections import defaultdict
import argparse
import copy
import math
from pathlib import Path
import random
import numpy as np
import torch

from features import H, mask_bins, forecast_inputs, build_features
from grounded_route import evidence_fields, jepa_fields
from route_heads import load_heads, route_logits
from train_v8 import restore_heads, loss, publish, load_partition


@torch.no_grad()
def blender(rows,ground,heads):
    out=[]
    for r in rows:
        text,ctx,local,a,coll=(r[k] for k in ("text","ctx","local","action","collision"))
        qs=r["queries"]
        if ctx.shape!=(12,3,1024) or local.shape!=(12,49,1024) or a.shape!=(2,24) or len(qs)!=2:
            raise ValueError("expected a two-query, two-route paired-layout state")
        code=(local-heads[0].center)@heads[0].basis
        anc=local@ground.anchors.T;sim=local@text.T
        v,n=ground(code.half().float(),anc.half().float(),sim.half().float(),text)
        for h in H:
            acts=a[:,:h]
            # Same prior view mask for the two swapped layouts, not derived from side or target.
            mask_key="controlled/"+r["house"]
            g=dict(h=h,sid=r["sid"],house=r["house"],kind=r["mode"],swap=bool(r["swap"]),family="blender",
                   query=qs,domain=["controlled"]*len(qs),y=torch.tensor([r["gain_by_query_by_horizon"][str(h)][q] for q in qs]).float().T,
                   mask=(acts[:,None]!=acts[None,:]).any(-1),matched=(acts[:,None]!=acts[None,:]).any(-1),modes=[],
                   progress=torch.stack([1.-(coll[:,:t]&(acts[:,:t]==0)).sum(1)/(acts[:,:t]==0).sum(1).clamp_min(1)
                                        for t in (h//2,h)],1),
                   forward=torch.stack([(acts[:,:t]==0).any(1) for t in (h//2,h)],1))
            for k in (12,1,2,3):
                bins=mask_bins(mask_key,k);yaw=torch.tensor(bins).float()*math.pi/6
                future,near,_=forecast_inputs(heads,"full" if k==12 else "sparse",ctx[bins],yaw,acts)
                dense,geo=evidence_fields(v[bins],n[bins],yaw,acts)
                g["modes"].append((dense,jepa_fields(future,near,text,ground.anchors),geo,
                                   jepa_fields(near,near,text,ground.anchors),route_logits(future,text).sigmoid()))
            out.append(g)
    return out


@torch.inference_mode()
def score(heads,groups):
    by=defaultdict(lambda:defaultdict(list)); detail={}
    for g in groups:
        for mi,(d,je,geo,near,v6) in enumerate(g["modes"]):
            arms={k:m(d,je,geo)[0].sigmoid() for k,m in heads.items()}
            arms.update(v6=v6,wrong=arms["joint"].flip(1),displacement=geo[:,-1,5,None].expand_as(v6),
                        joint_no_future=heads["joint"](d,near,geo)[0].sigmoid())
            for arm,p in arms.items():
                y=g["y"];delta_y=y[0]-y[1];s=p[0]-p[1];valid=delta_y!=0
                score=s*delta_y.sign()
                correct=(score>1e-7).float()+.5*(score.abs()<=1e-7)
                switched=s.abs()>.01
                gain=(torch.where(switched,s.sign()*delta_y*.5,0.)).mean().item()
                key=(g["kind"],g["h"],arm)
                by[key][g["house"]].append((correct[valid].sum().item(),int(valid.sum()),gain))
                if g["h"]==24 and mi==0:
                    detail[(g["kind"],g["house"],arm,g["swap"])]=s.tolist()
    out={}
    for (kind,h,arm),houses in by.items():
        rs=[];gs=[];swaps=[]
        for house,xs in houses.items():
            a,b,c=list(zip(*xs))
            if sum(b):rs.append(sum(a)/sum(b))
            gs.append(np.mean(c))
            if h==24:
                a,b=[np.array(detail[kind,house,arm,k]) for k in (False,True)]
                swaps.append(float(np.mean(np.sign(a)*np.sign(b)<0)))
        item=dict(houses=len(houses),equal_house_accuracy=float(np.mean(rs)) if rs else None,
                  equal_house_gated_gain=float(np.mean(gs)),full_view_pref_flips_on_swap=float(np.mean(swaps)) if swaps else None)
        out.setdefault(f"{kind}/{h}",{})[arm]=item
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("checkpoint","full","sparse","train","held","replay","output"):
        ap.add_argument("--"+k,type=Path,required=True)
    args=ap.parse_args()
    ground,heads,names=restore_heads(args.checkpoint)
    base=load_heads(args.full,args.sparse)
    a,n,replay_states=load_partition(args.replay)
    if n!=names or any(s["family"]!="primitives" for s in replay_states):
        raise ValueError("provide original natural primitive training states")
    train=blender(torch.load(args.train,weights_only=True),ground,base)
    replay=[g for s in replay_states for g in build_features(base,ground,s,a)]
    positive=[g for g in train if g["kind"]=="congruent"]
    assert len({g["house"] for g in positive})==4
    adapted={k:copy.deepcopy(v) for k,v in heads.items()}
    optim={k:torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.015) for k,m in adapted.items()}
    rng=random.Random(101)
    for ep in range(10):
        rng.shuffle(positive)
        for off in range(0,len(positive),4):
            batch=positive[off:off+4]
            xs=[(g,rng.randrange(4)) for g in batch+rng.choices(replay,k=len(batch))]
            for k,m in adapted.items():
                cost=torch.stack([loss(m,g,i) for g,i in xs]).mean()
                reg=torch.stack([(v-w.detach()).square().mean() for v,w in zip(m.parameters(),heads[k].parameters())]).sum()
                optim[k].zero_grad(set_to_none=True);(cost+.5*reg).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),3,error_if_nonfinite=True);optim[k].step()
    args.output.mkdir(exist_ok=True,parents=True)
    ck=args.output/"controlled_adapter.pt"
    if ck.exists(): raise FileExistsError(ck)
    torch.save({k:v.state_dict() for k,v in adapted.items()},ck)
    hold=blender(torch.load(args.held,weights_only=True),ground,base)
    if {g["house"] for g in train}&{g["house"] for g in hold}: raise ValueError("overlapping layouts")
    publish(args.output/"controlled.json",dict(results=dict(frozen_v8=score(heads,hold),controlled_adaptation=score(adapted,hold))))


if __name__=="__main__":
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(101);main()
