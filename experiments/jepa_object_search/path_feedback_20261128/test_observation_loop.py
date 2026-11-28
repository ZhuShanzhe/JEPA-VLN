"""Choose without simulator labels. The separately launched simulator scores the selected paths."""
from collections import Counter
import json
from pathlib import Path
import os
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from checker import PathChecker
from sensed_repair import choose_with_observation
from replan import embedding_sketch,RepairScorer

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/"pose_horizon_20261128"
sys.path.insert(0,str(PRIOR))
from observation_head import HorizonCameraPredictor
from prior_cues_score import FrozenImageEncoder,visual_input


def decisions(acts,depth,ctx,pose,head,checker,scorer):
    p=head(ctx,acts,pose)
    v=dict(sketch=embedding_sketch(p["predicted_future"],ctx[:,:1024]),old_joint=p["p_joint"],
           p_visible_if_reach=p["p_visible_if_reach"])
    plans=[tuple(a[a>0].tolist()) for a in acts]
    budget=max(map(len,plans))+4
    result={}
    for name,arm,views in (("raw","original",1),("checker_only","gate_only",1),
           ("oneshot_1","one_shot",1),("trained_1","trained",1),("oneshot_3","one_shot",3),("trained_3","trained",3)):
        for mode,initial in (("straight",0),("jepa",int(np.argmax(v["old_joint"])))):
            view_index=iter(range(3))
            # Frames are exposed only as observe() is called; unrequested views never reach the actor.
            result[mode+"_"+name]=choose_with_observation(plans,v,lambda turns:depth[next(view_index)],
                checker,scorer,initial,budget,arm,views)
    return result


def main():
    os.umask(0o077)
    torch.set_num_threads(2)
    head=HorizonCameraPredictor.from_checkpoints(PRIOR/"private_train","cuda" if torch.cuda.is_available() else "cpu")
    checker=PathChecker.load(HERE/"private_checker/checker.joblib")
    scorer=RepairScorer(HERE/"private_repair/with_hint.pt")
    enc=FrozenImageEncoder(PRIOR.parent)
    exported=[]
    historical={}
    for file in sorted((HERE/"private_history").glob("*.npz")):
        with np.load(file) as z:
            e=enc.encode([visual_input(z["rgb"]),visual_input(z["crop"])]).ravel()
            a=z["action"]
            outputs=decisions(a,z["depth"],np.tile(e,(len(a),1)),z["pose"],head,checker,scorer)
            historical[file.stem]=outputs
    for file in sorted((HERE/"private_looks").glob("*.npz")):
        house=file.stem
        with np.load(HERE/"private_data"/("test_"+house+".npz")) as d,np.load(file) as obs:
            groups=d["group"];pid=d["plan_id"];a=d["action"];x=d["pose"];c=d["context"]
            for i,views in enumerate(obs["depth"]):
                for n in (4,8):
                    ix=np.flatnonzero(groups==f"{house}/{i}/d{n}")
                    ix=ix[np.argsort(pid[ix])]
                    assert np.array_equal(pid[ix],np.arange(7))
                    outputs=decisions(a[ix],views,c[ix],x[ix],head,checker,scorer)
                    exported.append(dict(house=house,cue=i,distance=n,decisions=outputs))
    private=HERE/"private_observe"
    private.mkdir(exist_ok=True)
    (private/"holdout_decisions.json").write_text(json.dumps(exported))
    (private/"historical_decisions.json").write_text(json.dumps(historical))
    counts={};lamp={}
    for name,outputs in historical.items():
        with np.load(HERE/"private_history"/(name+".npz")) as z:
            labels=z["labels"]
            for key,result in outputs.items():
                totals=counts.setdefault(key,Counter())
                ix=result["index"];v=result["view"]
                ok=blocked=pix=0;cost=result["sensing_actions"]
                if ix is not None:ok,blocked,pix,cost=map(int,labels[v,ix])
                totals.update(cues=1,issued=int(ix is not None),reached=ok,blocked=blocked,
                    reached_and_clear=int(ok and pix>=369),sensing_actions=result["sensing_actions"],all_actions=cost)
                if name=="1_1":
                    lamp[key]=dict(reached=bool(ok),target_pixels=pix if ix is not None else None,
                        status=result["status"],touched_obstacle=bool(blocked),actions=cost)
    summary=dict(scope="Same 1m forward observation waypoint for all four past cue boxes. Every final route and requested physical scan has separately executed Habitat outcomes. Known diagnostic cases. Original navigation previously allowed three of four cue locations to see the instance by staying.",counts=counts,lamp=lamp)
    (HERE/"historical_repair.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("Selected paths for",len(exported),"held-out cue/waypoint groups and",len(historical),"historical cases.")

if __name__=="__main__":
    with threadpool_limits(limits=2):main()
