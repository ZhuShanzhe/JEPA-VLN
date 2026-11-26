"""Train future re-detection utility under controlled and natural reference box errors."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"active"))
from train_views import displacement,rows
from active_view_score import ActiveViewScorer,motion_vector,source_vector
from weak_view_score import WeakViewScorer,context_features


SEED=2502


def observed(p):
    root=p/"jepa_reid_v21"
    sys.path.insert(0,str(root/"model"))
    from encode_proposals import ious,assignments
    frames=rows(root/"data/frames.jsonl")
    dets={r["frame_id"]:r["boxes"] for r in rows(root/"detector/predictions.jsonl")}
    ck=torch.load(root/"model/proposals.pt",weights_only=True,map_location="cpu")
    for path in (root/"data/frames.jsonl",root/"detector/predictions.jsonl"):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==ck["hashes"][path.name]
    latent=F.normalize(ck["features"][:,:1536].float()).numpy()
    proj=np.random.default_rng(SEED).standard_normal((latent.shape[1],48),dtype="f")
    reduced=latent@proj
    code={(r["frame"],r["slot"]):reduced[i] for i,r in enumerate(ck["records"]) if not r["jitter"]}
    seq=defaultdict(list);assigned={}
    for f in frames:
        fid=f["frame_id"];seq[f["sequence"]].append(f)
        f["camera_xyz"]=np.asarray(f["T_world_from_camera"])[:3,3]
        pred=np.asarray([b["xyxy"] for b in dets[fid]],dtype=float).reshape(-1,4)
        gt=np.asarray([b["xyxy"] for b in f["boxes"]],dtype=float).reshape(-1,4)
        pairs=assignments(pred,gt)
        assigned[fid]={f["boxes"][j]["instance"]:i for i,j in pairs.items()}
        f["overlap"]=ious(pred,gt) if len(gt) and len(pred) else np.empty((len(pred),len(gt)))
        f["det"]=pred;f["gt"]=gt;f["matched"]=set(pairs)
    datasets=defaultdict(list)
    for group in seq.values():
        for f in group:
            alternatives=[q for q in group if q is not f]
            if len(alternatives)!=2:
                continue
            fid=f["frame_id"]; im=np.array(Image.open(root/"data"/f["rgb"]).convert("RGB"))
            motion=np.array([displacement(f,alt) for alt in alternatives],dtype="f")
            a=motion_vector(motion)
            for j,obj in enumerate(f["boxes"]):
                instance=obj["instance"]; gt=f["gt"][j];ww,hh=gt[2:]-gt[:2]
                available={}
                if instance in assigned[fid]:
                    available["detector_matched"]=f["det"][assigned[fid][instance]]
                # Keep a natural unmatched detector region only for unambiguous overlap
                # with this source instance; GT is used for supervision and cohort labels.
                partial=[]
                for i,b in enumerate(f["det"]):
                    if i in f["matched"]:
                        continue
                    overlaps=f["overlap"][i]
                    if np.argmax(overlaps)!=j or not .05<=overlaps[j]<.5:
                        continue
                    other=np.delete(overlaps,j)
                    if other.size and overlaps[j]-max(other)<.05:
                        continue
                    partial.append(i)
                if partial:
                    k=max(partial,key=lambda i:f["overlap"][i,j])
                    available["detector_fragment"]=f["det"][k]
                rng=np.random.default_rng(int(hashlib.sha256((fid+instance).encode()).hexdigest()[:8],16))
                center=gt[:2]+np.array([ww,hh])*rng.uniform(.2,.8,2)
                size=np.maximum(2.,np.array([ww,hh])*rng.uniform(.22,.42,2))
                available["synthetic_partial"]=np.r_[center-size/2,center+size/2]
                # Deliberately displace the whole reference to nearby background when
                # a non-overlapping crop of the same dimensions fits on that side.
                for sign in rng.permutation([-1,1]):
                    bb=np.array([gt[0]+sign*ww,gt[1],gt[2]+sign*ww,gt[3]])
                    if bb[0]>=0 and bb[2]<=im.shape[1] and not np.any(ious([bb],f["gt"])>.05):
                        available["synthetic_offset"]=bb
                        break
                y=[];z=[];seen=[]
                for alt in alternatives:
                    aid=alt["frame_id"];slot=assigned[aid].get(instance)
                    seen.append(any(o["instance"]==instance for o in alt["boxes"]))
                    y.append(int(slot is not None))
                    z.append(code.get((aid,slot)) if slot is not None else None)
                for name,bb in available.items():
                    xraw=context_features(im,bb)
                    area=np.prod(bb[2:]-bb[:2])/np.prod(im.shape[:2])
                    datasets[f["split"]].append(dict(x=xraw,motion=motion,a=a,y=y,z=z,area=area,
                         h=f["house"],kind=name,seen=seen))
    return datasets


def arrays(groups):
    features=[];target=[];present=[];quality=[]
    for g in groups:
        features.append(np.column_stack([np.repeat(g["x"][None],2,axis=0),g["a"]]))
        quality.extend(g["y"])
        for z in g["z"]:
            present.append(z is not None)
            target.append(z if z is not None else np.zeros(48,dtype="f"))
    return np.concatenate(features).astype("f"),np.array(quality,"f"),np.stack(target),np.array(present)


def measure(groups,scores):
    if not groups:
        return {"groups":0}
    selected=[]
    for g,rank in zip(groups,np.reshape(scores,(-1,2))):
        both=np.isclose(rank,rank.max(),atol=1e-7)
        selected.append(float(np.mean(np.asarray(g["y"])[both])))
    return {"groups":len(groups),"houses":len(set(g["h"] for g in groups)),"redetection":float(np.mean(selected))}


class Head(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.fc=nn.Linear(n,96);self.lat=nn.Linear(96,48);self.read=nn.Linear(48,1)
    def forward(self,x):
        z=self.lat(F.relu(self.fc(x))); return self.read(F.relu(z)).flatten(),z


def run(p,out):
    torch.set_num_threads(2)
    sets=observed(p);out.mkdir(exist_ok=True,parents=True)
    train,val,test=[sets[k] for k in ("train","val","test")]
    houses=[set(g["h"] for g in s) for s in (train,val,test)]
    assert all(not houses[i]&houses[j] for i in range(3) for j in range(i))
    x,y,z,prior=arrays(train);xv,yv,_,_=arrays(val);xt,yt,_,_=arrays(test)
    mu=x.mean(0);std=x.std(0).clip(.015)
    zprior=z[prior]; z=(z-zprior.mean(0))/zprior.std(0).clip(.05)
    tensor=lambda a:torch.from_numpy(np.asarray(a))
    tr=tensor(((x-mu)/std).clip(-8,8));tv=tensor(((xv-mu)/std).clip(-8,8))
    y,z,prior=tensor(y),tensor(z),tensor(prior)
    result={"protocol":"independent matched/fragment/cropped/off-target RGB sources; target is subsequent one-to-one same-instance detector IoU >=0.5; cases overlap across corruption conditions; small is <=2% of image.",
            "training_future_embedding_valid":int(prior.sum()),"splits":{k:{"houses":len(set(g["h"] for g in s)),
            "two_option_cases":len(s),"groups_by_source":dict(Counter(g["kind"] for g in s))} for k,s in sets.items()},"calibration":{}}
    scores={}
    for name,weight in (("direct",0.),("jepa",.12)):
        torch.manual_seed(SEED);gen=torch.Generator().manual_seed(SEED)
        m=Head(x.shape[1]);opt=torch.optim.AdamW(m.parameters(),lr=.0007)
        best=-1
        for epoch in range(30):
            for idx in torch.randperm(len(y),generator=gen).split(96):
                q,zz=m(tr[idx]);ok=prior[idx]
                aux=F.smooth_l1_loss(zz[ok],z[idx][ok]) if ok.any() else 0.
                loss=F.binary_cross_entropy_with_logits(q,y[idx])+weight*aux
                opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():
                pred=m(tv)[0].numpy().reshape(-1,2)
            stats=[]
            for k in sorted(set(g["kind"] for g in val)):
                ii=[i for i,g in enumerate(val) if g["kind"]==k]
                stats.append(measure([val[i] for i in ii],pred[ii])["redetection"])
            v=np.mean(stats)
            if v>best:
                best=v;result["calibration"][name]=dict(epoch=epoch+1,source_macro_redetection=float(v))
                params={lab:t.detach().numpy() for lab,t in zip(("w1","b1","w2","b2","w3","b3"),m.parameters())}
                np.savez(out/(name+".npz"),mean=mu,scale=std,**params)
        scores[name]=WeakViewScorer(out/(name+".npz"))._predict(xt).reshape(-1,2)
    # The original clean-box head sees the same source portion and pose, without
    # the extra masked context stream. It remains an unchanged reference.
    for k in ("direct","jepa"):
        ck=p/"jepa_reid_v25/active/weights"/(k+".npz")
        scores["old_"+k]=ActiveViewScorer(ck)._predict(np.column_stack([xt[:,:101],xt[:,-8:]])).reshape(-1,2)
    move=np.array([g["motion"] for g in test])
    for name,geom in (("forward",move[:,:,0]),("minimum_turn",-abs(move[:,:,2]))):
        # Same geometric tie convention as the existing clean-reference benchmark.
        scores[name]=np.isclose(geom,geom.max(axis=1,keepdims=True),atol=1e-6).astype("f")
    scores.update(random=np.zeros_like(move[:,:,0]),oracle=yt.reshape(-1,2))
    result["test"]={}
    for kind in sorted(set(g["kind"] for g in test)):
        for label,small in (("all",False),("small",True)):
            select=[i for i,g in enumerate(test) if g["kind"]==kind and (not small or g["area"]<=.02)]
            gs=[test[i] for i in select]
            byhead={k:measure(gs,v[select]) for k,v in scores.items()}
            pair=defaultdict(list)
            for i,g in zip(select,gs):
                a=measure([g],scores["jepa"][i])["redetection"]
                b=measure([g],scores["direct"][i])["redetection"]
                pair[g["h"]].append(a-b)
            if pair:
                h=np.array([np.mean(v) for v in pair.values()])
                byhead["house_jepa_minus_direct"]=dict(mean=float(h.mean()),ci95=np.quantile(
                    np.random.default_rng(SEED).choice(h,(5000,len(h))).mean(1),[.025,.975]).tolist())
            result["test"][kind+"_"+label]=byhead
    (out/"weak_results.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:{a:round(x.get("redetection",0),3) for a,x in v.items() if a in ("jepa","direct","old_jepa","forward")} for k,v in result["test"].items()}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("project",type=Path);parser.add_argument("out",type=Path)
    a=parser.parse_args();run(a.project,a.out)
