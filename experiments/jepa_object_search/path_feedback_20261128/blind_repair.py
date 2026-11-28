"""Frozen candidate evaluation on six new building families. No fitting or threshold changes."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse
import json
import numpy as np
import torch
from train_repair import HERE,OLD,groups,embed,HorizonCameraPredictor
from replan import RepairScorer,decide


def screen_first(s,scorer,existing=(0,1,2),trained=True):
    idx=np.array(existing)
    ok=idx[s.decision[idx]=="accept"]
    product=s.reach*s.visible_if_reached
    if len(ok):return int(ok[np.argmax(product[ok])]),False
    current=0
    considered=set(existing)
    for _ in range(2):  # initial canonical seed plus at most two new proposals
        choices=np.array([i for i in range(len(s.length)) if i not in considered],int)
        if not len(choices):break
        scores=scorer.score(s,current) if trained else product
        current=int(choices[np.argmax(scores[choices])]);considered.add(current)
        if s.decision[current]=="accept":return current,True
    return None,True


def evaluate(checkname,expected,split="blind",shards=None):
    torch.set_num_threads(3)
    shards=shards if shards is not None else HERE/"private_blind_paired"
    folder=HERE/("private_"+split+"_repair");folder.mkdir(exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    jepa=HorizonCameraPredictor.from_checkpoints(OLD/"private_train",device)
    files=sorted(shards.glob(split+"_*.npz"))
    assert len(files)==expected and len(list((HERE/checkname).glob(split+"_*.npz")))==expected
    for p in files:embed(p,jepa,folder)
    scorers={a:RepairScorer(HERE/"private_repair"/(f+".pt")) for a,f in
             (("trained","with_hint"),("without_hint","without_hint"),("geometry_only","geometry_only"),("mask_at_test","with_hint"))}
    scorers["mask_at_test"].hints=False
    counts=defaultdict(Counter);by_house=defaultdict(lambda:defaultdict(Counter))
    for house,_,s,r,v,ids in groups(split,shards,HERE/checkname,folder):
        joint=r&v
        def record(name,i,initial,triggered=False):
            row=dict(groups=1,issued=int(i is not None),reached=int(i is not None and r[i]),
                     joint=int(i is not None and joint[i]),issued_blocked=int(i is not None and not r[i]),
                     reobserve=int(i is None),changed=int(i is not None and i!=initial),
                     nominal_actions=int(s.length[i]) if i is not None else 0,
                     oracle_reach=int(r.any()),oracle_joint=int(joint.any()),triggered=int(triggered),
                     trigger_recovered=int(triggered and i is not None and r[i]),
                     source_blocked_recoverable=int(not r[initial] and r.any()),
                     recovered=int(not r[initial] and i is not None and r[i]),
                     lost_reachable=int(r[initial] and (i is None or not r[i])))
            counts[name].update(row);by_house[house][name].update(row)
        for arm in ("original","gate_only","one_shot","independent","trained","without_hint","geometry_only","mask_at_test"):
            for start,initial in (("straight",0),("old_jepa",int(np.argmax(s.old_joint)))):
                op=decide(s,initial,max(s.length),arm,scorers.get(arm))
                record(start+"_"+arm,op["index"],initial)
        for name,trained in (("hybrid_three_conditioned",True),("hybrid_three_independent",False)):
            choice,trigger=screen_first(s,scorers["trained"],trained=trained)
            record(name,choice,0,trigger)
    assert len(by_house)==expected
    report=dict(houses=expected,counts=counts,by_house=[dict(counts=v) for _,v in sorted(by_house.items())],
                note="Seven physically labelled plans per fixed target/endpoint. Hybrid first screens direct and both narrow detours, then offers a maximum of two unseen plans following rejection. Seven-way one-shot and three-admission hybrid have different candidate screenings. No fitting on these data.")
    (HERE/(checkname.removeprefix("private_")+"_repair.json")).write_text(json.dumps(report,indent=2)+"\n")
    print({k:[v["issued"],v["reached"],v["joint"],v["issued_blocked"]] for k,v in counts.items()})

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("predictions",help="directory name of row-aligned frozen checker predictions")
    args=p.parse_args();evaluate(args.predictions,6)
