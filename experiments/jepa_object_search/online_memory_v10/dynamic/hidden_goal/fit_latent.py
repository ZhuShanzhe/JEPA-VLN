"""Learn an action-conditioned visual prior and freeze it before evaluating held-out houses."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def read(path):
    return json.loads(path.read_bytes())


def unit(x):
    return x / np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-9)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root",type=Path);ap.add_argument("experiment",type=Path)
    args=ap.parse_args()
    root,d=args.root.resolve(),args.experiment.resolve()
    base=root/"jepa_five_20261123"
    sys.path[:0]=list(map(str,(base/"report_v8",base/"report_v7",base/"clip_v5",
                                root/"mapping_runtime/pilots/jepa_round2_five_20261121")))
    from cache_observed import assets
    from fit_v8 import restore_heads
    from route_heads import load_heads
    from grounded_route import evidence_fields,jepa_fields
    from report_head import forecast_inputs

    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    enc,_=assets();clip=enc.encoder
    ground,nets,_=restore_heads(base/"report_v8/v8.pt")
    route=load_heads(base/"clip_v5/head_v5.pt",base/"sparse_v6/sparse_head.pt")
    qs=["cup","sofa"]
    text=clip.texts(qs);words=torch.stack([torch.from_numpy(text[q]) for q in qs])
    config=read(d/"assets/protocol.json")
    source_cache={};future_cache={};frozen_v8={}
    def obtain(path):
        return np.array(Image.open(path).convert("RGB"))
    def embed(parts):
        batch=torch.stack([clip.prep(Image.fromarray(np.ascontiguousarray(v))) for v in parts])
        return F.normalize(clip.net.encode_image(batch).float(),dim=-1).numpy()
    def features(folder,key):
        if key not in source_cache:
            a=obtain(folder/(key+".png"))
            # The two plans approach opposite sides of the fork. Their image windows
            # are fixed from the camera geometry, and require no target detector.
            sides=embed([a[8:216,0:128],a[8:216,128:256]])
            source_cache[key]=unit(sides-sides[::-1])
        return source_cache[key]
    def future(folder,keys):
        if keys[0] not in future_cache or keys[1] not in future_cache:
            imgs=[obtain(folder/(k+".png"))[100:256,62:194] for k in keys]
            for k,v in zip(keys,embed(imgs)):
                future_cache[k]=v
        return np.stack([future_cache[k] for k in keys]).mean(0)
    def v8scores(folder,key,actions):
        if key not in frozen_v8:
            img=obtain(folder/(key+".png"))
            loc=enc.local([img])
            code,anc,sim=enc.compressed(loc,words)
            v,n=ground(code.half().float(),anc.half().float(),sim.half().float(),words)
            ctx=F.normalize(torch.tensor(clip.images([img])).float(),dim=-1)
            acts=torch.tensor(actions);heading=torch.zeros(1)
            p,near,_=forecast_inputs(route,"sparse",ctx,heading,acts)
            dense,geom=evidence_fields(v,n,heading,acts)
            lat=jepa_fields(p,near,words,enc.anchors)
            frozen_v8[key]=nets["joint"](dense,lat,geom)[0].sigmoid().numpy()
        return frozen_v8[key]
    source=[];teacher=[];targets=[];weights=[];sids=[]
    with torch.inference_mode():
        for h in config["split"]["train"]:
            f=d/"raw"/f"house_{h:02}"
            for r in read(f/"manifest.json"):
                k=r["key"];a=read(f/"actor"/(k+".json"));t=read(f/"teacher"/(k+".json"))
                x=features(f,a["context"][0])
                z=np.stack([future(f,images) for images in t["future"]])
                source.extend(x);teacher.extend(unit(z-z[::-1]));targets.extend(np.array(t["gain"])*2-1)
                weights.extend([config["train_weight"][r["mode"]]]*2)
                sids.append(f.name+"/"+k)
            print("train",h,flush=True)
    X,Y,L,w=[np.asarray(v,np.float64) for v in (source,teacher,targets,weights)]
    w=w[:,None]**.5
    Xw=w*X
    # Dual ridge regression on fixed visual features is a small joint-embedding
    # predictor. Its two outputs are the teacher feature and a direct-label control.
    reg=1.
    system=Xw@Xw.T+np.eye(len(Xw))*reg
    alpha=np.linalg.solve(system,w*Y)
    direct=np.linalg.solve(system,w*L)
    prototype=unit(L.T@Y)
    model=d/"train_only_head.npz"
    if model.exists():
        with np.load(model) as old:
            for k,t in (("Xw",Xw),("alpha",alpha),("direct",direct),("prototype",prototype)):
                np.testing.assert_allclose(old[k],t,rtol=1e-8,atol=1e-10)
    else:
        np.savez_compressed(model,Xw=Xw,alpha=alpha,direct=direct,prototype=prototype)
    ck=hashlib.sha256(model.read_bytes()).hexdigest()
    with np.load(model) as locked:
        Xw,alpha,direct,prototype=[locked[n] for n in ("Xw","alpha","direct","prototype")]
    predictions={}
    with torch.inference_mode():
        for h in config["split"]["test"]:
            f=d/"raw"/f"house_{h:02}"
            for r in read(f/"manifest.json"):
                k=r["key"];a=read(f/"actor"/(k+".json"))
                # No teacher data or hidden images are read while scoring the test set.
                x=features(f,a["context"][0]);phi=x@Xw.T
                latent=phi@alpha
                scores=latent@prototype.T
                v8=v8scores(f,a["context"][0],a["plans"])
                predictions[f.name+"/"+k]={name:t.tolist() for name,t in dict(
                    latent_head=scores,direct_supervised=phi@direct,wrong_goal=scores[:,::-1],
                    pure_text=latent@words.numpy().T,frozen_v8=v8,equal_geometry=np.zeros((2,2))).items()}
            print("score",h,flush=True)
    (d/"predictions.json").write_text(json.dumps(predictions))
    groups={};per_house={};paired=0;pxmax=0;mincue=999999;route_dev=0.;hashes=[]
    for h in config["split"]["test"]:
        f=d/"raw"/f"house_{h:02}";per_house[f.name]={}
        for r in read(f/"manifest.json"):
            k=r["key"];t=read(f/"teacher"/(k+".json"))
            y=np.asarray(t["gain"],int);mode=r["mode"]
            pxmax=max(pxmax,max(t["context_target_max"]))
            if mode!="no_cues":mincue=min(mincue,min(t["context_cue_max"]))
            route_dev=max(route_dev,abs(t["travel"][0]-t["travel"][1]))
            other_mode={"aligned":"contradictory","contradictory":"aligned","no_cues":"no_cues"}[mode]
            other=f'{r["variant"]}_{other_mode}_{1-r["placement"]}'
            aa=read(f/"actor"/(k+".json"));bb=read(f/"actor"/(other+".json"))
            if aa["context"]!=bb["context"]:
                raise RuntimeError(f"Counterfactual target changed observed RGB in {f.name}/{k}")
            paired+=1
            hashes.append(hashlib.sha256(json.dumps([aa,t["gain"]],sort_keys=True).encode()).hexdigest())
            for name,scores in predictions[f.name+"/"+k].items():
                score=np.array(scores)
                diff=(score*(y*2-1)).sum(0)
                acc=(diff>1e-7).astype(float)+.5*(np.abs(diff)<=1e-7)
                groups.setdefault(mode,{}).setdefault(name,[]).extend(acc)
                per_house[f.name].setdefault(mode,{}).setdefault(name,[]).extend(acc)
    def mean_dict(source):
        return {mode:{name:float(np.mean(vals)) for name,vals in stats.items()} for mode,stats in source.items()}
    means=mean_dict(groups)
    means["cue_mixture_75_25"]={k:.75*v+.25*means["contradictory"][k] for k,v in means["aligned"].items()}
    means["full_factorial"]={k:float(np.mean([means[m][k] for m in ("aligned","contradictory","no_cues")])) for k in means["aligned"]}
    receipt=dict(train_houses=len(config["split"]["train"]),test_houses=len(config["split"]["test"]),train_starts=len(sids),
        test_starts=len(predictions),test_queries=2*len(predictions),pairwise_visibility_matches=paired//2,
        max_initial_target_pixels=pxmax,min_visible_cue_pixels=mincue,max_imbalance_m=route_dev,
        training_prior_weight=config["train_weight"],arm_means=means,house_accuracies={k:mean_dict(v) for k,v in per_house.items()},
        weights_sha256=ck,test_hash=hashlib.sha256("".join(hashes).encode()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation="Two categories; engineered visual cue-target correlation; six new procedurally generated room geometries. No natural-house performance claim.")
    (d/"hidden_summary.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps(means,indent=2),flush=True)


if __name__=="__main__":main()
