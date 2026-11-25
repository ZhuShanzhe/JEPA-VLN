"""Frozen paired choices at a common physical action budget for every method and query."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F


def comparison(houses, ours, other):
    delta = ours.astype(int) - other.astype(int)
    per_house = np.array([delta[houses == h].mean() for h in sorted(set(houses))])
    sampled = np.random.default_rng(1701).choice(per_house, (20000,len(per_house)),replace=True).mean(axis=1)
    return dict(our_only=int((delta>0).sum()), their_only=int((delta<0).sum()),
                both_hit=int((ours & other).sum()), both_miss=int((~ours & ~other).sum()),
                raw_count_delta=int(delta.sum()), house_macro_difference=float(per_house.mean()),
                percentile_95=np.quantile(sampled, [.025,.975]).tolist())


def run(project):
    stage = project / "jepa_variable_v16/model_training"
    old = project / "jepa_evidence_v15/train"
    sys.path[:0] = [str(stage),str(old),str(project/"jepa_hidden_v11/models")]
    from train_prefix import distribution, frozen_options
    from train import load
    from evidence_head import EvidenceNet
    keys, data = load(project,"fresh2",False)
    houses = np.array([h for h,_,_ in keys])
    truth = []
    for house,origin,word in keys:
        with np.load(stage/"private_fresh"/(origin+".npz")) as z:
            assert str(z["house"]) == house
            truth.append(z["visible"][z["words"].tolist().index(word)])
    truth = np.stack(truth)
    assert np.array_equal(truth[:,:,5::6].any(-1),data["y"]>0)
    features,goals = [torch.from_numpy(data[k]).float() for k in ("x","text")]
    calibration=json.loads((project/"jepa_variable_v16/model/calibration.json").read_text())
    freeze=json.loads((stage/"freeze.json").read_text())
    original=json.loads((stage/"equal_cost.json").read_text())["results"]
    assert set(houses).isdisjoint(freeze["training_houses"] + freeze["calibration_houses"])
    arms={"jepa":(stage/"jepa.pt",freeze["choices"]["jepa"]["sha256"]),
          "direct":(stage/"direct.pt",freeze["choices"]["direct"]["sha256"]),
          "production_evidence_rate":(old/"frozen.pt",calibration["checkpoint_sha256"]),
          "v15_jepa_same_utility":(old/"frozen.pt",calibration["checkpoint_sha256"])}
    logits={}
    for name,(path,expected_hash) in arms.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected_hash
        net=EvidenceNet(goals.size(-1))
        net.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
        net.eval()
        with torch.inference_mode():
            logits[name]=torch.cat([net(x,g)[1] for x,g in zip(features.split(96),goals.split(96))])
    results={}
    for n in (2,3):
        _,ids=frozen_options(keys,n)
        for steps in (6,12,18,24):
            taps=steps//6
            hits={}
            d=truth[:,:,:steps].any(-1)
            for name,z in logits.items():
                if name=="production_evidence_rate":
                    v=-torch.expm1(F.logsigmoid(-(calibration["scale"]*z[:,:,:taps]+calibration["offset"])).mean(-1))
                else:
                    v=distribution(z,torch.full(z.shape[:2],taps))[1]
                chosen=ids.gather(1,v.gather(1,ids).argmax(1,keepdim=True)).numpy().flatten()
                hits[name]=d[np.arange(len(keys)),chosen]
                if name in original:
                    assert int(hits[name].sum())==original[name][str(n)][str(steps)]["selected"]
            available=d[np.arange(len(keys))[:,None],ids.numpy()]
            results[f"{n}routes_{steps}actions"]=dict(actions_per_query=steps,total_actions_per_method=len(keys)*steps,
                oracle_any=int(available.any(1).sum()),uniform_expected=float(available.mean(1).sum()),
                counts={k:int(v.sum()) for k,v in hits.items()}, paired_jepa_minus={
                    k:comparison(houses,hits["jepa"],v) for k,v in hits.items() if k!="jepa"})
    return dict(unit="one query with target visible on any step of its one chosen path",
                split="HSSD fresh2",houses=len(set(houses)),queries=len(keys),origins=len(set(k[1] for k in keys)),
                interval="paired equal-house mean differences; 20000 percentile bootstrap draws over houses, seed 1701",
                results=results)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project",type=Path)
    parser.add_argument("destination",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    data=run(args.project.resolve())
    content=json.dumps(data,indent=2,allow_nan=False)+"\n"
    if args.destination.exists() and args.destination.read_text()!=content:
        raise FileExistsError(args.destination)
    args.destination.write_text(content)
    print(json.dumps({k:v for k,v in data["results"].items() if "24actions" in k}))
