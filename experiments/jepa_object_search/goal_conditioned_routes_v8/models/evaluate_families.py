"""Evaluate two candidate families on the same houses and exact mirrored motion pairs."""
from collections import defaultdict
import argparse
from pathlib import Path
import torch
import numpy as np

from train_v8 import restore_heads, load_partition, evaluate, publish, permutation
from route_heads import load_heads
from features import build_features

@torch.inference_mode()
def mirror_metrics(networks,groups):
    cells=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for g in groups:
        perm=permutation(g["query"],g["domain"],g["sid"],g["h"])
        i,j=torch.triu(g["mirror"],diagonal=1).nonzero(as_tuple=True)
        if not len(i):continue
        for mi,(dense,je,geo,near,v6) in enumerate(g["modes"]):
            arms={k:m(dense,je,geo)[0].sigmoid() for k,m in networks.items()}
            nd,nj=g["null_modes"][mi]
            arms.update(wrong=arms["joint"][:,perm],no_word=networks["joint"](nd,nj,geo)[0].sigmoid().expand_as(v6),
                        displacement=geo[:,-1,5,None].expand_as(v6),v6=v6,
                        joint_no_future=networks["joint"](dense,near,geo)[0].sigmoid())
            for d in ("official","derived","all"):
                idx=[k for k,v in enumerate(g["domain"]) if d=="all" or v==d]
                y=g["y"][:,idx]; gap=y[i]-y[j]
                ab,qa,qb=((gap[:,:,None]>0)&(gap[:,None,:]<0)).nonzero(as_tuple=True)
                for name,scores in arms.items():
                    p=scores[:,idx];delta=p[i]-p[j]
                    mask=gap!=0
                    a=delta*gap.sign()
                    hit=((a>1e-7)+.5*(a.abs()<=1e-7))[mask].sum().item()
                    first,second=delta[ab,qa],delta[ab,qb]
                    contrast=first-second
                    credit=((contrast>1e-7)+.5*(contrast.abs()<=1e-7)).sum().item()
                    strict=((first>1e-7)&(second< -1e-7)).sum().item()
                    cells[g["family"],d,name][g["house"]][g["sid"]].append(
                        (hit,int(mask.sum()),credit,strict,len(ab),len(i),len(idx)))
    out={}
    for (family,d,name),houses in cells.items():
        means=[[],[],[]];starts=[0,0];pairs=0;reversals=0
        for by_start in houses.values():
            hm=[[],[],[]]
            for rows in by_start.values():
                agg=np.asarray(rows).sum(0);pairs+=agg[1];reversals+=agg[4]
                if agg[1]:hm[0].append(agg[0]/agg[1]);starts[0]+=1
                if agg[4]:
                    hm[1].append(agg[2]/agg[4]);hm[2].append(agg[3]/agg[4]);starts[1]+=1
            for k,c in enumerate(hm):
                if c:means[k].append(np.mean(c))
        vals=[float(np.mean(v)) if v else None for v in means]
        out.setdefault(family+"|"+d,{})[name]=dict(rank=vals[0],double_difference=vals[1],
              both_target_choices_correct=vals[2],rank_starts=starts[0],opposite_starts=starts[1],
              informative_houses=len(means[0]),opposite_houses=len(means[1]),
              informative_goal_pair_entries_including_view_repeats=int(pairs),
              opposed_goal_pair_entries_including_view_repeats=int(reversals))
    return out



def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("checkpoint","full","sparse","states","output"):
        ap.add_argument("--"+k,required=True,type=Path)
    arg=ap.parse_args()
    ground,nets,words=restore_heads(arg.checkpoint)
    anchors,names,samples=load_partition(arg.states)
    if words!=names: raise ValueError("misaligned anchor bank")
    if {s["family"] for s in samples}!={"primitives","frontiers"}:
        raise ValueError("the family comparison requires both action families")
    rooms={family:{s["house"] for s in samples if s["family"]==family} for family in ("primitives","frontiers")}
    if rooms["primitives"]!=rooms["frontiers"]: raise ValueError("families must share the same houses")
    base=load_heads(arg.full,arg.sparse)
    groups=[g for s in samples for g in build_features(base,ground,s,anchors)]
    publish(arg.output,dict(metrics=evaluate(nets,groups),exact_mirror=mirror_metrics(nets,groups)))


if __name__=="__main__":
    torch.set_num_threads(2); main()
