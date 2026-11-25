"""Fit direct and predictive matchers on an identical house-disjoint pair protocol."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from reid_model import Encoder, OBJECT_DIM, REPRESENTATION, ViewHead, mask_context, preprocess


SEED=3161
FPR=0.05


class Crops(Dataset):
    def __init__(self, root, rows):
        self.root,self.rows=root,rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r=self.rows[index]
        with Image.open(self.root/r["image"]) as f:
            obj=preprocess(np.asarray(f.convert("RGB")))
        with Image.open(self.root/r["context_image"]) as f:
            ctx=preprocess(mask_context(np.asarray(f.convert("RGB")),r["box_in_context"]))
        return obj,ctx


def extract(project,manifest,rows,out,device):
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    path=out/"features.pt"
    if path.exists():
        saved=torch.load(path,weights_only=True)
        assert saved["manifest_sha256"]==digest and saved["representation"]==REPRESENTATION
        return saved["features"]
    model=Encoder(project).to(device).eval()
    dataset=Crops(manifest.parent,rows)
    loader=DataLoader(dataset,batch_size=24,shuffle=False,num_workers=4,pin_memory=(device=="cuda"))
    features=[]
    for index,(crops,ctx) in enumerate(loader):
        features.append(model(crops.to(device),ctx.to(device)).cpu())
        if index%32==0:
            print("encoded",len(features)*24,"of",len(rows),flush=True)
    features=torch.cat(features)
    torch.save(dict(features=features,manifest_sha256=digest,representation=REPRESENTATION),path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)
    return features


def delta(rows,i,j):
    return (rows[j]["camera_heading_deg"]-rows[i]["camera_heading_deg"]+180)%360-180


def make_pairs(rows,split):
    rng=random.Random(SEED)
    groups=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(rows):
        if r["split"]==split:
            groups[(r["house"],r["category"])][r["instance"]].append(i)
    pairs=[]
    for groupset in groups.values():
        if len(groupset)<2:
            continue
        for key,ids in groupset.items():
            other=[k for name,members in groupset.items() if name!=key for k in members]
            for i in ids:
                pos=[j for j in ids if j!=i and abs(delta(rows,i,j))>=15.]
                rng.shuffle(pos)
                negative_pool=other.copy()
                rng.shuffle(negative_pool)
                for j in pos[:5]:
                    if not negative_pool:
                        break
                    # Match the imposed rotation, preventing gross camera-angle shortcuts on negative pairs.
                    k=min(negative_pool,key=lambda k:abs(delta(rows,j,k)))
                    negative_pool.remove(k)
                    pairs.extend([(i,j,1),(i,k,0)])
    if not pairs:
        raise ValueError("No matched within-house same-category pairs in "+split)
    return np.array(pairs,dtype=np.int64)


def choose_threshold(y,s):
    fp,tp,thr=roc_curve(y,s,drop_intermediate=False)
    indexes=np.flatnonzero(fp<=FPR+1e-12)
    chosen=indexes[np.argmax(tp[indexes])]
    return float(thr[chosen]) if np.isfinite(thr[chosen]) else float(np.nextafter(s.max(),np.array(np.inf,dtype=s.dtype)))


def describe(y,s,thr):
    y=y.astype(bool)
    pred=s>=thr
    def ratio(k,n):
        return float(k/n) if n else None
    return dict(positive=int(y.sum()),negative=int((~y).sum()),
                tp=int((pred & y).sum()),fp=int((pred & ~y).sum()),
                tpr=ratio((pred & y).sum(),y.sum()), fpr=ratio((pred & ~y).sum(),(~y).sum()),
                auroc=float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,
                average_precision=float(average_precision_score(y,s)) if y.any() else None)


@torch.no_grad()
def predict(head,feat,pairs,angle,device):
    scores=[]
    for start in range(0,len(pairs),512):
        p=pairs[start:start+512]
        a=angle[start:start+512].to(device)
        logits,_=head(feat[p[:,0]],feat[p[:,1]],a)
        scores.append(logits.sigmoid().cpu())
    return torch.cat(scores).numpy()


def fit(mode,features,ps,angle,device,out):
    torch.manual_seed(SEED)
    model=ViewHead().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.015)
    train=ps["train"]
    val=ps["val"]
    best=-1
    history=[]
    for epoch in range(18):
        model.train()
        idx=np.random.default_rng(SEED+epoch).permutation(len(train))
        for pos in range(0,len(idx),256):
            batch=idx[pos:pos+256]
            p=train[batch]
            label=torch.from_numpy(p[:,2]).to(device).float()
            logits,sim=model(features[p[:,0]],features[p[:,1]],angle["train"][batch].to(device))
            loss=F.binary_cross_entropy_with_logits(logits,label)
            if mode.startswith("jepa") and (label>0).any():
                # Both target values are outputs of the frozen encoder. Each forecast
                # can use only the source representation and the camera rotation.
                loss=loss+3.0*(1.-sim[label>0].mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
            optimizer.step()
        model.eval()
        scores=predict(model,features,val,angle["val"],device)
        pvc=float(roc_auc_score(val[:,2],scores,max_fpr=FPR))
        history.append(pvc)
        if pvc>best:
            best=pvc
            ck=dict(model=mode,representation=REPRESENTATION,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    epoch=epoch+1,threshold=choose_threshold(val[:,2],scores),preset_false_positive_rate=FPR)
    torch.save(ck,out/(mode+".pt"))
    model.load_state_dict(ck["state_dict"])
    return model,history


def main(args):
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in args.manifest.read_text().splitlines() if x]
    sets={name:{r["house"] for r in rows if r["split"]==name} for name in ("train","val","test")}
    assert all(sets.values()) and sum(map(len,sets.values()))==len(set.union(*sets.values()))
    device="cuda" if torch.cuda.is_available() else "cpu"
    feats=extract(args.project,args.manifest,rows,out,device).to(device)
    ps={name:make_pairs(rows,name) for name in sets}
    angles={name:torch.tensor([delta(rows,i,j) for i,j,_ in p],dtype=torch.float32) for name,p in ps.items()}
    result=dict(protocol="Both directions; all impostors same category in the same house. Validation threshold caps empirical FPR at 5%. Results count pairs, not independent episodes.",seed=SEED,
                image_count=len(rows),houses={k:len(v) for k,v in sets.items()},device=device,
                pair_counts={k:len(v) for k,v in ps.items()},thresholds={},metrics={},validation_curves={})
    scores={k:{} for k in ("val","test")}
    trained={"direct","jepa","jepa_object"}
    for mode in ["frozen","object","context","direct","jepa","jepa_object"]:
        mode_feats=feats
        if mode=="jepa_object":
            mode_feats=feats.clone()
            mode_feats[:,OBJECT_DIM:]=0.
            mode_feats=F.normalize(mode_feats,dim=-1)
        if mode in trained:
            head,curve=fit(mode,mode_feats,ps,angles,device,out)
            result["validation_curves"][mode]=curve
        for split in scores:
            p=ps[split]
            if mode in trained:
                score=predict(head,mode_feats,p,angles[split],device)
            else:
                chunks=[]
                dims={"frozen":slice(None),"object":slice(OBJECT_DIM),"context":slice(OBJECT_DIM,None)}[mode]
                for i in range(0,len(p),512):
                    ix=p[i:i+512]
                    chunks.append(F.cosine_similarity(feats[ix[:,0],dims],feats[ix[:,1],dims]).cpu())
                score=torch.cat(chunks).numpy()
            scores[split][mode]=score
        thr=choose_threshold(ps["val"][:,2],scores["val"][mode])
        result["thresholds"][mode]=thr
        result["metrics"][mode]={}
        for split in scores:
            p=ps[split]; y=p[:,2]; score=scores[split][mode]
            table=describe(y,score,thr)
            ad=np.abs(angles[split].numpy())
            table["angle_bins"]={f"{lo}_{hi}":describe(y[m],score[m],thr) for lo,hi in [(0,30),(30,60),(60,100),(100,181)] if (m:=((ad>=lo)&(ad<hi))).any()}
            positive=y.astype(bool)
            same_asset=np.array([rows[i].get("asset")==rows[j].get("asset") and rows[i].get("asset") is not None for i,j in p[:,:2]])
            table["asset_matched_impostors"]=describe(y[(~positive)&same_asset],score[(~positive)&same_asset],thr)
            if "context" in scores["val"] and "context" in scores[split]:
                vc=scores["val"]["context"][ps["val"][:,2]==1]
                cut=float(np.median(vc))
                weak_context=scores[split]["context"]<cut
                table["positive_low_context_similarity"]=describe(y[positive&weak_context],score[positive&weak_context],thr)
                table["positive_high_context_similarity"]=describe(y[positive&~weak_context],score[positive&~weak_context],thr)
                table["context_split_validation_median"]=cut
            if mode in trained and split=="test":
                zero_yaw_scores=predict(head,mode_feats,p,torch.zeros_like(angles[split]),device)
                table["heading_zero_diagnostic_at_original_threshold"]=describe(y,zero_yaw_scores,thr)
                with torch.no_grad():
                    positive_indices=np.flatnonzero(positive)
                    future_cos,static_cos=[],[]
                    for ix in np.array_split(positive_indices,max(1,int(np.ceil(len(positive_indices)/512)))):
                        pi=p[ix]
                        source,target=mode_feats[pi[:,0]],mode_feats[pi[:,1]]
                        forecast=head.forecast(source,angles[split][ix].to(device))
                        future_cos.append((forecast*target).sum(-1).cpu())
                        static_cos.append((source*target).sum(-1).cpu())
                    table["positive_forecast_cosine"]=float(torch.cat(future_cos).mean())
                    table["positive_static_cosine"]=float(torch.cat(static_cos).mean())
            table["by_house"]={}
            for house in sorted(sets[split]):
                m=np.array([rows[i]["house"]==house for i in p[:,0]])
                if m.any():
                    table["by_house"][house]=describe(y[m],score[m],thr)
            result["metrics"][mode][split]=table
        print(mode,"test",result["metrics"][mode]["test"]["tpr"],result["metrics"][mode]["test"]["fpr"],flush=True)
    for split,score_dict in scores.items():
        p=ps[split]
        np.savez_compressed(out/("pairs_"+split+".npz"),i=p[:,0],j=p[:,1],y=p[:,2],**{"s_"+k:v for k,v in score_dict.items()})
    (out/"metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for opt in ("project","manifest","out"):
        parser.add_argument("--"+opt,type=Path,required=True)
    main(parser.parse_args())
