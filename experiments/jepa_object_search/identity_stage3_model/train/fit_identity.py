"""Fit paired verification objectives on development houses and freeze validation thresholds."""
import argparse
from collections import Counter,defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F


SEED=126111
EPOCHS,UPDATES=12,48
QUOTAS={"orbit_positive":32,"orbit_negative":32,"natural_positive":32,"natural_negative":32,
        "twins_positive":32,"twins_negative":32}


def checksum(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def collect(project,out):
    old,base=project/"jepa_reid_v20",project/"jepa_reid_v21"
    sys.path[:0]=[str(old/"model"),str(base/"model")]
    from train import make_pairs
    from train_pose import legacy_rotations
    manifest=old/"data/manifest.jsonl"
    raw=list(map(json.loads,manifest.read_text().splitlines()))
    dev={"train","val"}
    dst={s:defaultdict(list) for s in dev}
    houses={s:{r["house"] for r in raw if r["split"]==s} for s in ("train","val","test")}
    assert all(not (houses[a]&houses[b]) for a,b in combinations(houses,2))
    fields=[[],[],[],[]]
    used=[manifest,old/"model/features.pt",base/"model/geometry_all.npz",base/"model/proposals.pt"]
    cache=torch.load(used[1],map_location="cpu",weights_only=True)
    geometry=np.load(used[2],allow_pickle=False)
    assert cache["manifest_sha256"]==geometry["manifest_sha256"].item()==checksum(manifest)
    assert np.array_equal(geometry["ix"],np.arange(len(raw)))
    keep=np.array([i for i,r in enumerate(raw) if r["split"] in dev])
    assert geometry["valid"][keep].all()
    small=[raw[i] for i in keep]
    fields[0].append(cache["features"][keep])
    fields[1].append(legacy_rotations(small))
    fields[2].append([r["camera_xyz"] for r in small])
    fields[3].append(geometry["xyz"][keep])
    for s in dev:
        for i,j,t in make_pairs(small,s):
            dst[s]["orbit_"+("positive" if t else "negative")].append((i,j,t))
    start=len(small)
    def append_cache(cache,kind):
        nonlocal start
        rec=cache["records"]
        keep=[i for i,r in enumerate(rec) if r["split"] in dev and r["instance"] and not r["jitter"]]
        assert all(rec[i]["house"] in houses[rec[i]["split"]] for i in keep)
        selected=[rec[i] for i in keep]
        fields[0].append(cache["features"][keep])
        for field,key in zip(fields[1:],("R","camera","point")):
            field.append([r[key] for r in selected])
        groups=defaultdict(list)
        for i,row in enumerate(selected):
            group=(row["split"],row["house"],row["category"])
            if kind=="twins":
                group+=(row["sequence"].rsplit("_",1)[0],)
            groups[group].append(i)
        for (s,*_),indices in groups.items():
            for i,j in combinations(indices,2):
                a,b=selected[i],selected[j]
                t=int(a["instance"]==b["instance"])
                if a["frame"]==b["frame"] and t:
                    continue
                dst[s][kind+("_positive" if t else "_negative")].append((start+i,start+j,t))
        start+=len(selected)
    append_cache(torch.load(used[3],map_location="cpu",weights_only=True),"natural")
    for split in ("train","val"):
        for name in (split, split+"_clocks"):
            path=out/(name+".pt")
            c=torch.load(path,map_location="cpu",weights_only=True)
            assert set(r["split"] for r in c["records"])=={split}
            for file in [out/name/"frames.jsonl",out/name/"det/predictions.jsonl"]:
                assert c["hashes"][file.name]==checksum(file)
            append_cache(c,"twins")
            used.append(path)
    features=torch.cat(fields[0]).cuda()
    dataset=[features,*[torch.tensor(np.concatenate(v),device="cuda",dtype=torch.float32) for v in fields[1:]]]
    pairs={s:{k:np.asarray(v,dtype=np.int64) for k,v in groups.items()} for s,groups in dst.items()}
    assert all(set(g)==set(QUOTAS) for g in pairs.values())
    return dataset,pairs,houses,{str(p.relative_to(project)):checksum(p) for p in used}


def cutoff(y,s,fpr):
    neg=np.sort(s[y==0])
    max_count=int(np.floor(fpr*len(neg)))
    return float(np.nextafter(neg[-(max_count+1)],np.float32(np.inf)))


def fit(project,seed=SEED,output=None):
    training_root=project/"identity_stage3_model/train"
    out=training_root/"private" if output is None else output
    out.mkdir(parents=True,exist_ok=True)
    if (out/"jepa.pt").exists() or (out/"direct.pt").exists():
        raise FileExistsError("select a fresh output directory; retain frozen weights")
    dataset,pairs,houses,hashes=collect(project,training_root/"private")
    from train_pose import score_batch,scores
    from pose_head import PoseHead
    valgroups={}
    for origin in ("orbit","natural","twins"):
        valgroups[origin]=np.concatenate([pairs["val"][origin+"_"+key] for key in ("positive","negative")])
    report=dict(train_houses=len(houses["train"]),validation_houses=len(houses["val"]),seed=seed,
        training=dict(epochs=EPOCHS,updates_per_epoch=UPDATES,batch_counts=QUOTAS,
                      rules="same architecture/labels/order/perturbations; JEPA adds positive pair frozen-feature prediction loss"),
        pairs={s:{k:len(v) for k,v in g.items()} for s,g in pairs.items()},hashes=hashes,methods={})
    for mode in ("direct","jepa"):
        torch.manual_seed(seed)
        rng=np.random.default_rng(seed)
        head=PoseHead().cuda()
        opt=torch.optim.AdamW(head.parameters(),lr=.00065,weight_decay=.01)
        best=-np.inf
        for epoch in range(EPOCHS):
            head.train()
            for _ in range(UPDATES):
                p=np.concatenate([pairs["train"][k][rng.integers(len(pairs["train"][k]),size=n)] for k,n in QUOTAS.items()])
                p=p.copy()
                flip=rng.random(len(p))<.5
                p[flip,:2]=p[flip,:2][:,::-1]
                raw,sim=score_batch(head,"pose_jepa",dataset,p,training=True)
                t=torch.tensor(p[:,2],dtype=torch.float32,device="cuda")
                loss=F.binary_cross_entropy_with_logits(raw,t)
                if mode=="jepa":
                    loss=loss+3*(1-sim[t.bool()].mean())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),3.)
                opt.step()
            head.eval()
            preds={k:scores(head,"pose_jepa",dataset,v) for k,v in valgroups.items()}
            value=np.mean([roc_auc_score(valgroups[k][:,2],v,max_fpr=.05) for k,v in preds.items()])
            if value>best:
                best=value
                ck=dict(mode=mode,epoch=epoch+1,head={k:t.detach().cpu().clone() for k,t in head.state_dict().items()})
                best_scores=preds
        # Hard impostors and natural negatives must each satisfy the calibration
        # false-acceptance aim. Separately limit false "different" on positives.
        high=max(cutoff(valgroups[k][:,2],v,.05) for k,v in best_scores.items())
        low=min(-cutoff(1-valgroups[k][:,2],-v,.05) for k,v in best_scores.items())
        low=min(low,float(np.nextafter(np.float32(high),-np.float32(np.inf))))
        ck.update(high=high,low=low,margin=.05)
        temp=out/(mode+".tmp")
        torch.save(ck,temp)
        temp.replace(out/(mode+".pt"))
        report["methods"][mode]=dict(selected_epoch=ck["epoch"],high=high,low=low,margin=.05,
            parameter_count=sum(p.numel() for p in head.parameters()),validation_macro_pAUC_5pct=float(best),
            weights_sha256=checksum(out/(mode+".pt")),validation={k:dict(
                pairs=len(v),same_recall=float(np.mean(v[valgroups[k][:,2]==1]>=high)),
                impostor_accept=float(np.mean(v[valgroups[k][:,2]==0]>=high)),
                reject_true_same=float(np.mean(v[valgroups[k][:,2]==1]<=low))) for k,v in best_scores.items()})
        print(mode,report["methods"][mode],flush=True)
    for f in ("identity.py","fit_identity.py","training_twins.py","clock_seeds.py"):
        report["hashes"][f]=checksum(training_root/f)
    for f in ["jepa_reid_v21/model/train_pose.py","jepa_reid_v21/model/pose_head.py","jepa_reid_v20/model/reid_model.py","jepa_reid_v20/model/train.py","jepa_reid_v21/model/encode_proposals.py","mapping_runtime/weights/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"]:
        report["hashes"][f]=checksum(project/f)
    (training_root/"training_results.json" if output is None else out/"training_results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("project",type=Path)
    ap.add_argument("--seed",type=int,default=SEED)
    ap.add_argument("--output-dir",type=Path)
    args=ap.parse_args()
    fit(args.project.resolve(),args.seed,args.output_dir)
