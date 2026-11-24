"""Compare frozen predictive embeddings, within-origin route identity, and navigation decisions."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-10)


def evaluate_prediction(pred, future, actions):
    p,t=unit(pred),unit(future)
    n=len(actions)
    # Cross-route similarities keep the house, start, step and image region matched.
    dots=np.einsum("athd,bthd->abt",p,t,optimize=True)/p.shape[2]
    table=dots.mean(axis=-1)
    correct=np.diag(table)
    greatest=table.max(axis=1,keepdims=True)
    tied=(greatest-table)<=1e-6
    credit=np.diag(tied).astype(float)/tied.sum(axis=1)
    # The sign of this double difference tests the correct two-route assignment.
    # It cancels high global similarity of a particular room or default future.
    assignment=[];margins=[];by_tap=[[] for _ in range(t.shape[1])]
    for i,j in zip(*np.triu_indices(n,1)):
        if np.array_equal(actions[i],actions[j]):continue
        if float(np.max(np.abs(t[i]-t[j])))<1e-6:continue
        delta=dots[i,i]+dots[j,j]-dots[i,j]-dots[j,i]
        diff=float(delta.mean());assignment.append(.5 if abs(diff)<=1e-6 else float(diff>0));margins.append(diff)
        for k,a in enumerate(delta):
            h=round((k+1)*actions.shape[1]/t.shape[1])
            if np.array_equal(actions[i,:h],actions[j,:h]):continue
            if np.max(np.abs(t[i,k]-t[j,k]))<1e-6:continue
            by_tap[k].append(.5 if abs(a)<=1e-6 else float(a>0))
    def avg(a):return float(np.mean(a)) if len(a) else None
    return dict(cosine_error=float(1.-correct.mean()), route_top1=float(credit.mean()),
            chance_top1=1./n, pair_assignment=avg(assignment),assignment_margin=avg(margins),
            **{f"pair_tap_{k+1}":avg(a) for k,a in enumerate(by_tap)})


def macro(values):
    if not values:return None
    return float(np.mean(values))


def corr(x,y):
    if len(x)<2 or np.std(x)<1e-10 or np.std(y)<1e-10:return None
    return float(np.corrcoef(x,y)[0,1])


@torch.inference_mode()
def run(root,outfile):
    sys.path.insert(0,str(root/"models"))
    from train_compare import Ranker,route_inputs,SEEDS
    d=root/"models";folder=d/"cache/hssd/test"
    locked=json.loads((d/"fit_receipt.json").read_bytes());models={}
    for mode,config in locked.items():
        for r in config["runs"]:
            arm,seed=r["arm"],r["seed"]
            file=d/"weights"/mode/f"{arm}_{seed}.pt"
            assert hashlib.sha256(file.read_bytes()).hexdigest()==r["checkpoint_sha256"]
            ck=torch.load(file,map_location="cpu",weights_only=True); m=Ranker(ck["goal.weight"].shape[1])
            m.load_state_dict(ck);models[mode,arm,seed]=m.eval()
    assert len(models)==18
    nav=defaultdict(dict)
    for l in (d/"test_predictions.jsonl").read_text().splitlines():
        r=json.loads(l)
        if r["source"]!="hssd":continue
        nav[(r['house'],r['origin'],r['query'],r['mode'],r['seed'])][r['arm']]=r
    if not nav:raise ValueError("No frozen navigation predictions")
    torch.set_num_threads(2)
    records=defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
    query_opps={}
    for file in sorted((folder/"actor").glob("*.npz")):
        with np.load(file) as z:
            ctx,action=z["ctx"].astype("float32"),z["actions"]
            house,origin=str(z["house"]),str(z["origin"])
            words=z["words"].tolist();txt=torch.tensor(z["text"][0:1])
        x=torch.tensor(route_inputs(ctx,action))[None]
        with np.load(folder/"teacher"/file.name) as z: future=z["future"].astype("float32")
        key=(house,origin)
        yaw=np.cumsum((action==1).astype(float)-(action==2),axis=1)*30.
        taps=[round(frac*action.shape[1])-1 for frac in (.25,.5,.75,1.)]
        chosen=np.cos(np.deg2rad(yaw[:,taps,None]-np.array([-30.,0.,30.]))).argmax(-1)
        baseline=evaluate_prediction(ctx[chosen],future,action)
        for (mode,arm,seed),model in models.items():
            _,prediction=model(x,txt)
            metric=evaluate_prediction(prediction[0].numpy(),future,action)
            good=[]
            for word in words:
                row=nav[house,origin,word,mode,seed][arm]
                gain=np.asarray(row["gain"])
                if (gain>0).any() and (gain<=0).any():
                    good.append(float(gain[row["first"]]>0))
            metric["first_informative_hit"]=macro(good)
            query_opps[key]=len(good)
            records[mode][key][arm][seed]=metric
            records[mode][key]["copy_start_view"][seed]=baseline
    # Average seeds at each actual start before averaging starts and houses.
    output={};association={};diffs={}
    for mode,items in records.items():
        summaries=defaultdict(dict)
        for key,arms in items.items():
            for arm,seeds in arms.items():
                assert set(seeds)==set(SEEDS)
                for metric in next(iter(seeds.values())):
                    values=[v[metric] for v in seeds.values() if v[metric] is not None]
                    if not values:continue
                    summaries[arm,metric][key]=float(np.mean(values))
        def houses(values):
            group=defaultdict(list)
            for (h,_),v in values.items():group[h].append(v)
            return {h:float(np.mean(v)) for h,v in group.items()}
        output[mode]=defaultdict(dict)
        for (arm,m),values in summaries.items():output[mode][arm][m]=macro(list(houses(values).values()))
        diffs[mode]={}
        for ctl in ("direct","wrong_future","copy_start_view"):
            diffs[mode][ctl]={}
            for metric in ("cosine_error","route_top1","pair_assignment"):
                base=summaries[ctl,metric];alt=summaries["jepa",metric]
                delta=houses({k:alt[k]-v for k,v in base.items() if k in alt})
                dd=np.array(list(delta.values()))
                ci=np.quantile(np.random.default_rng(701).choice(dd,(10000,len(dd)),replace=True).mean(1),[.025,.975])
                diffs[mode][ctl][metric]=dict(houses=len(dd),mean=float(dd.mean()),paired_bootstrap_95=ci.tolist())
        nav_delta={k:v-summaries['direct','first_informative_hit'][k] for k,v in summaries['jepa','first_informative_hit'].items()}
        association[mode]={}
        for metric,sign in (("cosine_error",-1),("pair_assignment",1)):
            better={k:sign*(summaries['jepa',metric][k]-summaries['direct',metric][k]) for k in nav_delta if k in summaries['jepa',metric]}
            a=np.array(list(better.values()));y=np.array([nav_delta[k] for k in better])
            pos=a>1.e-6
            association[mode][metric]=dict(origins=len(a),pearson_descriptive=corr(a,y),
                    representation_improved=int(sum(pos)),nav_better_with_rep_improved=int(sum(pos&(y>1e-6))),
                    nav_tied_with_rep_improved=int(sum(pos&(abs(y)<=1e-6))),
                    nav_worse_with_rep_improved=int(sum(pos&(y< -1e-6))))
    result=dict(houses=len({k[0] for k in query_opps}),origins=len(query_opps),
        origins_with_variable_binary_outcomes=sum(v>0 for v in query_opps.values()),
        goal_origins_with_variable_binary_outcomes=sum(query_opps.values()),seeds=list(SEEDS),
        alignment="Mean cosine distance across four times and three regions. Route top1 compares candidates only at the same start, two-way assignment cancels shared background.",
        macro=output,jepa_minus_comparison=diffs,representation_navigation_association=association)
    outfile.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":run(Path(sys.argv[1]),Path(sys.argv[2]))
