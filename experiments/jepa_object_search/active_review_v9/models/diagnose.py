"""Evaluate fair-look controls and previously reported known-instance counterfactuals."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from fit import C,D,teacher,features,history_features,write,y_by_term
from encode_data import encode
from v9_core import Local,ReviewHead,descriptors
from v9_live import Features


def matched(p,y,valid,cut):
    a=np.broadcast_to(np.arange(12)[:,None],(12,len(valid))).copy()
    start=a.copy();qi=np.broadcast_to(np.arange(len(valid)),a.shape);hi=np.ones_like(a)
    init=y[a,qi,0].astype(bool); seen=init.copy();on=np.broadcast_to(valid,a.shape).copy()
    spent=np.zeros_like(a)
    for _ in range(2):
        score=p[a,hi,qi]
        on&=(score[...,0]<cut)&(score[...,1:].max(-1)>=.15)&(score[...,1:].max(-1)-score[...,0]>.03)
        sign=np.where(score[...,1]>=score[...,2],1,-1)
        a=(a+sign*on)%12; hi=np.where(on,1-sign,hi);spent+=on
        seen|=on&y[a,qi,0].astype(bool)
    sides=[]
    for sign in (1,-1):
        visited=init.copy()
        for step in (1,2):
            visited|=(spent>=step)&y[(start+step*sign)%12,qi,0].astype(bool)
        sides.append(visited)
    hard=~init&valid[None,:]
    return dict(total=int(np.broadcast_to(valid,a.shape).sum()),initially_missing=int(hard.sum()),spent=int(spent.sum()),
        with_any_turn=int((spent>0).sum()),learned=int((hard&seen).sum()),passive=0,
        left=int((hard&sides[0]).sum()),right=int((hard&sides[1]).sum()),
        random_expected=float(((hard&sides[0]).sum()+(hard&sides[1]).sum())/2),
        oracle=int((hard&(sides[0]|sides[1])).sum()))


def summarize(v):
    return dict(v,accuracy_where_oracle=v["learned"]/max(1,v["oracle"]),
                random_accuracy_where_oracle=v["random_expected"]/max(1,v["oracle"]))


def run():
    torch.set_num_threads(3)
    words=json.loads((D/"words.json").read_text());cut=json.loads((D/"policy.json").read_text())["threshold"] or 1.1
    rows=sorted(json.loads((C/"holdout.index.json").read_bytes()),key=lambda a:a["sid"])
    # The cold-start predictions and primary result are sealed before these diagnostics.
    assert (D/"blind_results.json").is_file()
    tables=defaultdict(lambda:defaultdict(float));mem=defaultdict(lambda:defaultdict(int))
    bank=defaultdict(dict)
    enc=Features(D);encode(enc,"holdout",True);del enc
    ck=torch.load(D/"v9.pt",weights_only=True);local=Local(len(words)).eval();head=ReviewHead().eval()
    local.load_state_dict(ck["local"]);head.load_state_dict(ck["head"])
    with torch.inference_mode():
        for row in rows:
            t=teacher(row,words);y=y_by_term(t,len(words));prov=t["provenance"]
            f=D/"blind_predictions"/(hashlib.sha256(row["sid"].encode()).hexdigest()+".npz")
            with np.load(f) as z:p=z["prob"]
            for group in ["all",prov,"house/"+row["house"]]:
                for name,num in matched(p,y,t["valid"],cut).items():tables[group][name]+=num
            c,s=features(row);v,n=local(c,s)
            for q in np.flatnonzero(t["valid"]):
                opts=[i for i in t["ids"][t["classes"]==q] if i in bank[row["house"]]]
                if not opts:continue
                k=hashlib.sha256((row["sid"]+"|"+str(q)).encode()).digest()[0]
                iid=opts[k%len(opts)]; x=descriptors(v[:,:,q:q+1],n[:,:,q:q+1],s[:,:,q:q+1],c,bank[row["house"]][iid])
                scores=head.layers(history_features(x,0,True)).sigmoid()[:,0,0].numpy()
                before=p[:,1,q,0];old_y=y[:,q,0].astype(bool)
                new_y=y_by_term(t,len(words),iid)[:,q,0].astype(bool)
                for key,mask in [("duplicate_only",old_y&~new_y),("still_novel",new_y),("no_new_evidence",~old_y)]:
                    mem[key]["views"]+=int(mask.sum())
                    mem[key]["with_memory_alerts"]+=int(((scores>=cut)&mask).sum())
                    mem[key]["without_memory_alerts"]+=int(((before>=cut)&mask).sum())
            with np.load(D/"memory"/(row["sid"]+".npz")) as z:
                for i,c in zip(z["ids"],z["vectors"]):bank[row["house"]][i]=torch.from_numpy(c.copy())
    houses={k:f"house_{i}" for i,k in enumerate(sorted(k for k in tables if k.startswith("house/")),1)}
    write(D/"diagnostics.json",dict(matched_turns={houses.get(k,k):summarize(dict(v)) for k,v in tables.items()},conditional_memory=dict(mem)))


if __name__=="__main__":run()
