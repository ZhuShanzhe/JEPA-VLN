"""Predict future frozen DINO regions and rank counterfactual observed views in HSSD."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from active_view_score import ActiveViewScorer, motion_vector, source_vector


SEED=2501
EPOCHS=30


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def displacement(a,b):
    yaw=math.radians(a["camera_heading_deg"])
    v=np.asarray(b["camera_xyz"])-a["camera_xyz"]
    # Habitat XZ plane: heading atan2(world_z,world_x); right turns increase heading.
    return [v[0]*math.cos(yaw)+v[2]*math.sin(yaw),-v[0]*math.sin(yaw)+v[2]*math.cos(yaw),
            math.radians((b["camera_heading_deg"]-a["camera_heading_deg"]+180)%360-180)]


def data(p):
    folder=p/"jepa_reid_v20/data"
    manifest=folder/"manifest.jsonl"
    s=rows(manifest)
    cache=torch.load(p/"jepa_reid_v20/model/features.pt",map_location="cpu",weights_only=True)
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==cache["manifest_sha256"]
    feat=F.normalize(cache["features"][:,:1536].float()).numpy()
    assert len(s)==len(feat)
    # A fixed projection provides small stopped-gradient visual targets without fitting held-out features.
    rng=np.random.default_rng(SEED)
    projection=rng.standard_normal((feat.shape[1],48),dtype=np.float32)
    targets=feat@projection
    pools=defaultdict(list)
    x=[]
    byhouse={}
    for i,r in enumerate(s):
        assert byhouse.setdefault(r["house"],r["split"])==r["split"]
        pools[(r["house"],r["category"])].append(i)
        im=np.asarray(Image.open(folder/r["context_image"]).convert("RGB"))
        x0,y0,u,v=r["box_in_context"]
        x.append(source_vector(im[y0:v,x0:u],r["bbox"],(384,384)))
    x=np.array(x)
    output=defaultdict(list)
    for (house,category),idx in pools.items():
        local=feat[idx]@feat[idx].T
        instances=np.array([s[i]["instance"] for i in idx])
        for ii,src in enumerate(idx):
            peers=np.nonzero(instances!=instances[ii])[0]
            if not len(peers):
                continue
            siblings=np.nonzero(instances==instances[ii])[0]
            for jj in siblings:
                j=idx[jj]
                if j==src:
                    continue
                m=displacement(s[src],s[j])
                # Distances in this offline set reflect endpoints, not a simulated action path.
                if np.hypot(*m[:2])>3 or abs(s[j]["pitch_deg"]-s[src]["pitch_deg"])>20:
                    continue
                y=s[j]
                x0,y0,u,v=y["bbox"]
                purity=y["pixels"]/max(1,(u-x0)*(v-y0))
                size=min(1.,math.sqrt(y["pixels"])/80.)
                margin=local[ii,jj]-float(local[peers,jj].max())
                identity=1/(1+np.exp(-12*margin))
                quality=size*math.sqrt(purity)*identity
                output[y["split"]].append((src,j,m,quality,identity,house,category))
    return s,x,targets,output


def arrays(x,targets,items):
    i,j,m,q,identity,h,c=zip(*items)
    return dict(input=np.column_stack([x[list(i)],motion_vector(m)]).astype("f"),
                target=targets[list(j)],quality=np.array(q,dtype="f"),identity=np.array(identity),
                group=np.array(i),house=np.array(h),cat=np.array(c),motion=np.asarray(m))


class Model(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.fc=nn.Linear(width,96)
        self.latent=nn.Linear(96,48)
        self.score=nn.Linear(48,1)

    def forward(self,x):
        z=self.latent(F.relu(self.fc(x)))
        return self.score(F.relu(z)).squeeze(1),z


def grouped(dataset,scores,mask=None):
    buckets=defaultdict(list)
    for i,g in enumerate(dataset["group"]):
        if mask is None or mask[i]:
            buckets[g].append(i)
    out=[]
    for g,items in buckets.items():
        if len(items)<2:
            continue
        i=items[np.argmax(scores[items])]
        q=dataset["quality"][items]
        out.append((dataset["house"][i],dataset["quality"][i],np.mean(q),np.max(q),dataset["identity"][i]))
    return out


def metrics(dataset,scores):
    r=grouped(dataset,scores)
    if not r:
        return {}
    q=np.array([v[1:] for v in r])
    return dict(groups=len(r),houses=len(set(v[0] for v in r)),mean_selected=float(q[:,0].mean()),
                mean_random=float(q[:,1].mean()),mean_oracle=float(q[:,2].mean()),
                oracle_hit=float(np.isclose(q[:,0],q[:,2]).mean()),mean_selected_identity=float(q[:,3].mean()))


def main(args):
    torch.set_num_threads(3)
    p=args.project
    out=args.output
    out.mkdir(exist_ok=True,parents=True)
    s,source,z,sets=data(p)
    ds={k:arrays(source,z,v) for k,v in sets.items()}
    train,val,test=[ds[k] for k in ("train","val","test")]
    summary=dict(splits={},results={},weights_selected_by="highest mean selected calibration utility")
    house_sets=[set(a["house"]) for a in (train,val,test)]
    assert all(not (house_sets[i]&house_sets[j]) for i in range(3) for j in range(i))
    for name,a in ds.items():
        summary["splits"][name]=dict(transitions=len(a["quality"]),houses=len(set(a["house"])),
                                     anchor_observations=len(set(a["group"])),categories=len(set(a["cat"])))
    mu=train["input"].mean(0); sd=train["input"].std(0).clip(.015)
    zm=train["target"].mean(0); zd=train["target"].std(0).clip(.05)
    xt=torch.from_numpy(((train["input"]-mu)/sd).clip(-8,8))
    zz=torch.from_numpy(((train["target"]-zm)/zd).clip(-6,6))
    yy=torch.from_numpy(train["quality"])
    xv=torch.from_numpy(((val["input"]-mu)/sd).clip(-8,8))
    for name,weight in (("direct",0.),("jepa",.045)):
        torch.manual_seed(SEED)
        gen=torch.Generator().manual_seed(SEED)
        model=Model(xt.shape[1]); opt=torch.optim.AdamW(model.parameters(),lr=.0015)
        best=-1.; ck=out/(name+".npz")
        for epoch in range(EPOCHS):
            for inds in torch.randperm(len(yy),generator=gen).split(256):
                pred,latent=model(xt[inds])
                loss=F.mse_loss(pred,yy[inds])+weight*F.smooth_l1_loss(latent,zz[inds])
                opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():
                score=model(xv)[0].numpy()
            metric=metrics(val,score)["mean_selected"]
            if metric>best:
                best=metric; best_epoch=epoch+1
                params={lab:v.detach().numpy() for lab,v in zip(("w1","b1","w2","b2","w3","b3"),model.parameters())}
                np.savez(ck,mean=mu,scale=sd,**params)
        scorer=ActiveViewScorer(ck)
        summary["results"][name]={"epoch":best_epoch,"val":metrics(val,scorer._predict(val["input"])),
                                   "test":metrics(test,scorer._predict(test["input"]))}
    summary["results"]["smallest_turn"]={"test":metrics(test,-np.abs(test["motion"][:,2]))}
    summary["results"]["forward_displacement"]={"test":metrics(test,test["motion"][:,0])}
    jl=grouped(test,ActiveViewScorer(out/"jepa.npz")._predict(test["input"]))
    dl=grouped(test,ActiveViewScorer(out/"direct.npz")._predict(test["input"]))
    delta=defaultdict(list)
    for j,d in zip(jl,dl):
        delta[j[0]].append(j[1]-d[1])
    a=np.array([np.mean(v) for v in delta.values()])
    res=np.random.default_rng(SEED).choice(a,(5000,len(a))).mean(1)
    summary["paired_house_macro_jepa_minus_direct"]=dict(mean=float(a.mean()),ci95=np.quantile(res,[.025,.975]).tolist())
    (out/"training_results.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("project",type=Path)
    parser.add_argument("output",type=Path)
    main(parser.parse_args())
