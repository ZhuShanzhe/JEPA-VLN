"""Train a small context verifier on top of a frozen cross-view JEPA predictor."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn.functional as F


@torch.inference_mode()
def evidence(head, anchors, candidates, angles):
    future = head.forecast(anchors, angles)
    reverse = head.forecast(candidates, -angles)
    # Each context image has its object box masked. These comparisons can be
    # obtained from observable RGB and relative heading at inference.
    obj, ctx = slice(0, 1536), slice(1536, None)
    def sim(a,b,part):
        return F.cosine_similarity(a[:,part],b[:,part])
    obj_sim = sim(anchors,candidates,obj)
    ctx_sim = sim(anchors,candidates,ctx)
    obj_forecast = (sim(future,candidates,obj)+sim(reverse,anchors,obj))*.5
    ctx_forecast = (sim(future,candidates,ctx)+sim(reverse,anchors,ctx))*.5
    rotation = torch.sin(torch.deg2rad(angles)*.5).abs()
    return torch.stack([obj_sim,ctx_sim,obj_forecast,ctx_forecast,
                        obj_sim*ctx_sim,obj_forecast*ctx_forecast,
                        rotation,ctx_sim*rotation,obj_forecast*rotation],dim=1)


class Refiner(torch.nn.Module):
    def __init__(self, head, params):
        super().__init__()
        self.head = head
        for k in ("mean","std","coef","intercept"):
            self.register_buffer(k,torch.tensor(params[k],dtype=torch.float32))

    def forward(self,anchors,candidates,angles):
        x = evidence(self.head,anchors,candidates,angles)
        score = ((x-self.mean)/self.std) @ self.coef + self.intercept
        return score,None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    args = p.parse_args()
    saved = args.root / "model"
    sys.path.insert(0,str(saved))
    import train
    from reid_model import ViewHead
    torch.set_num_threads(2)
    rows = [json.loads(line) for line in (args.root/"data"/"manifest.jsonl").read_text().splitlines()]
    f = torch.load(saved/"features.pt",map_location="cpu",weights_only=True)["features"].float()
    ck = torch.load(saved/"jepa.pt",map_location="cpu",weights_only=True)
    head = ViewHead().eval()
    head.load_state_dict(ck["state_dict"])
    tr = train.make_pairs(rows,"train")
    pairs = {s:np.load(saved/f"pairs_{s}.npz",allow_pickle=False) for s in ("val","test")}
    for_calibration = {s:np.stack([p["i"],p["j"],p["y"]],axis=1) for s,p in pairs.items()}
    def get_features(p):
        out = []
        for k in range(0,len(p),256):
            a,b,_ = p[k:k+256].T
            ang = torch.tensor([train.delta(rows,i,j) for i,j in zip(a,b)],dtype=torch.float32)
            out.append(evidence(head,f[a],f[b],ang).numpy())
        return np.vstack(out).astype(np.float64)
    x = get_features(tr)
    mean,std = x.mean(axis=0),x.std(axis=0).clip(.001)
    y = tr[:,2]
    same_template = np.array([rows[a]["asset"] == rows[b]["asset"] for a,b,_ in tr])
    # Half of negative training weight on distinct physical copies of the same CAD.
    # Neither template nor physical instance ID is exposed to the verifier.
    masks = [y==1,(y==0) & same_template,(y==0) & ~same_template]
    w = np.zeros(len(y))
    for fraction,mask in zip((.5,.25,.25),masks):
        assert mask.any()
        w[mask] = len(y)*fraction/mask.sum()
    clf = LogisticRegression(C=.1,solver="lbfgs",max_iter=400,random_state=7)
    clf.fit((x-mean)/std,y,sample_weight=w)
    params = {"mean":mean.tolist(),"std":std.tolist(),"coef":clf.coef_[0].tolist(),
              "intercept":float(clf.intercept_[0])}
    out = args.root/"evaluation"
    (out/"refiner_parameters.json").write_text(json.dumps(params)+"\n")
    model = Refiner(head,params).eval()
    result = {"training_pair_counts":{"same_instance":int(masks[0].sum()),"distinct_same_asset":int(masks[1].sum()),
                                      "distinct_different_asset":int(masks[2].sum())}}
    threshold = None
    for s,p in for_calibration.items():
        a = torch.tensor([train.delta(rows,i,j) for i,j,_ in p],dtype=torch.float32)
        score = train.predict(model,f,p,a,"cpu")
        y = p[:,2]
        if s == "val":
            threshold = train.choose_threshold(y,score)
            result["pair_threshold"] = threshold
        result[s] = train.describe(y,score,threshold)
        hard = np.array([rows[i]["asset"] == rows[j]["asset"] for i,j,_ in p]) & (y == 0)
        result[s]["same_template_different"] = int(hard.sum())
        result[s]["same_template_wrongly_merged"] = int((hard & (score >= threshold)).sum())
    (out/"refiner_results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
