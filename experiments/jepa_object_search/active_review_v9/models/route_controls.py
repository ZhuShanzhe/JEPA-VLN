"""Direction controls, target strata and house-sensitive results for a frozen policy."""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from fit import C,D,teacher,write,y_by_term,trials,fmt


def fixed_budget(p,y,valid,cut):
    start=np.broadcast_to(np.arange(12)[:,None],(12,len(valid))).copy()
    query=np.broadcast_to(np.arange(len(valid)),start.shape)
    angle=start.copy();hist=np.ones_like(angle); on=np.broadcast_to(valid,angle.shape).copy()
    spent=np.zeros_like(angle);hidden=(y[start,query,0]==0)&on
    for _ in range(2):
        s=p[angle,hist,query];on&=(s[:,:,0]<cut)&(s[:,:,1:].max(-1)>=.15)&(s[:,:,1:].max(-1)-s[:,:,0]>.03)
        d=np.where(s[:,:,1]>=s[:,:,2],1,-1);angle=(angle+on*d)%12;hist=np.where(on,1-d,hist);spent+=on
    groups={}
    idx=np.flatnonzero(valid)
    wrong=np.arange(len(valid));wrong[idx]=np.roll(idx,1)
    for name in ["learned","single_frame","wrong_word"]:
        angle=start.copy();hist=np.ones_like(angle);seen=np.zeros_like(on)
        for t in range(2):
            on=spent>t
            s=p[angle,1 if name=="single_frame" else hist,wrong[query] if name=="wrong_word" else query]
            d=np.where(s[:,:,1]>=s[:,:,2],1,-1)
            angle=(angle+on*d)%12;hist=np.where(on,1-d,hist)
            seen|=on & y[angle,query,0].astype(bool)
        groups[name]=int((hidden&seen).sum())
    return groups


def main():
    words=json.loads((D/"words.json").read_text())
    rows=json.loads((C/"holdout.index.json").read_text());pfile=D/"policy.json"
    cut=json.loads(pfile.read_text())["threshold"] or 1.1
    per=defaultdict(lambda:np.zeros(9,int));ctrl=defaultdict(Counter)
    for r in rows:
        t=teacher(r,words);y=y_by_term(t,len(words))
        f=D/"blind_predictions"/(hashlib.sha256(r["sid"].encode()).hexdigest()+".npz")
        with np.load(f) as z:p=z["prob"]
        g=fixed_budget(p,y,t["valid"],cut)
        for group in ("all",t["provenance"],"house/"+r["house"]):
            ctrl[group].update(g)
        for j,word in enumerate(words):
            valid=np.arange(len(words))==j
            per[word]+=trials(p,y,valid&t["valid"],cut,"learned")
        per["house/"+r["house"]]+=trials(p,y,t["valid"],cut,"learned")
    rename={k:f"house_{j}" for j,k in enumerate(sorted(a for a in ctrl if a.startswith("house/")),1)}
    write(D/"route_controls.json",dict(equal_learned_turn_budget={rename.get(k,k):dict(v) for k,v in ctrl.items()},
         learned_metrics_by_class_and_house={rename.get(k,k):fmt(v) for k,v in per.items()}))


if __name__=="__main__":main()
