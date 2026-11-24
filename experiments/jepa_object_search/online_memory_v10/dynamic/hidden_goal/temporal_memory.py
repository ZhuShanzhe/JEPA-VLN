"""Train the scene-evidence reader on the eight training rooms and apply its frozen parameters."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def load(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-9)


def endpoints(plans,coord,heading):
    a=np.array(plans)
    yaw=math.atan2(heading[1],heading[0])+np.cumsum((a==1).astype(float)-(a==2).astype(float),axis=1)*math.pi/6
    directions=np.stack([np.cos(yaw),np.sin(yaw)],axis=-1)
    return coord+np.sum(directions*(a==0)[:,:,None]*.25,axis=1),directions[:,-1]


class Encoder:
    def __init__(self,clip):
        self.clip=clip;self.cache={}

    def embed(self,requests):
        missing={k:p for k,p in requests.items() if k not in self.cache}
        keys=list(missing)
        with torch.inference_mode():
            for j in range(0,len(keys),16):
                group=keys[j:j+16];batch=[]
                for key in group:
                    crop,_,_=key.partition(":")
                    a=np.array(Image.open(missing[key]).convert("RGB"))
                    if crop=="left":a=a[8:216,0:128]
                    elif crop=="right":a=a[8:216,128:256]
                    elif crop=="verify":a=a[100:256,62:194]
                    else: raise ValueError(crop)
                    batch.append(self.clip.prep(Image.fromarray(np.ascontiguousarray(a))))
                z=F.normalize(self.clip.net.encode_image(torch.stack(batch)).float(),dim=-1).numpy()
                self.cache.update(zip(group,z))


def prior_feature(encoder,path):
    digest=path.stem
    keys=["left:"+digest,"right:"+digest]
    encoder.embed({k:path for k in keys})
    f=np.stack([encoder.cache[k] for k in keys])
    return unit(f-f[::-1])


def margins(prior,features):
    kernel=features@prior["Xw"].T
    return np.stack([(kernel@prior["alpha"])@prior["prototype"].T,kernel@prior["direct"]])


def train(d,encoder):
    source=d/"raw";prior=np.load(d/"train_only_head.npz")
    requests={};targets={};scale=[]
    for h in load(d/"assets/protocol.json")["split"]["train"]:
        f=source/f"house_{h:02}"
        for r in load(f/"manifest.json"):
            a=load(f/"actor"/(r["key"]+".json"));t=load(f/"teacher"/(r["key"]+".json"))
            scale.append(np.abs(margins(prior,prior_feature(encoder,f/(a["context"][0]+".png")))))
            # Supervision for the small verification head. Each end-view category is
            # assigned from pixel visibility at exactly this step (no hidden actors).
            labels=[(a["context"][0],2),(a["context"][3],2),(a["context"][6],2),(a["context"][9],2)]
            for imgs,px in zip(t["future"],t["end_target_pixels"]):
                valid=np.where(np.array(px)>=64)[0]
                if len(valid)!=1:raise ValueError("Training endpoint has ambiguous visible instances")
                labels.append((imgs[-1],int(valid[0])))
            for key,label in labels:
                if key in targets and targets[key]!=label:raise ValueError("Conflicting RGB training label")
                targets[key]=label;requests["verify:"+key]=f/(key+".png")
    encoder.embed(requests)
    x=np.stack([encoder.cache["verify:"+k] for k in targets]).astype(np.float64)
    cls=np.array(list(targets.values()));count=np.bincount(cls,minlength=3)
    sample=np.sqrt(len(cls)/(3*count[cls]))[:,None]
    center=x.mean(axis=0);basis=unit(x-center)
    output=np.eye(3)[cls]
    xw=sample*basis
    alpha=np.linalg.solve(xw@xw.T+np.eye(len(xw))*.75,sample*output)
    scale=np.maximum(np.median(np.stack(scale),axis=(0,2,3)),1e-6)
    ck=d/"temporal"/"train_only_evidence.npz"
    if not ck.exists():np.savez_compressed(ck,xw=xw,alpha=alpha,center=center,scale=scale)
    else:
        with np.load(ck) as old:
            for k,arr in dict(xw=xw,alpha=alpha,center=center,scale=scale).items():
                np.testing.assert_allclose(old[k],arr,atol=1e-9)
    print("TRAINED",dict(zip(("cup","sofa","background"),count.tolist())),sha(ck),flush=True)


def score(d,encoder):
    temporal=d/"temporal";base=temporal/"streams"
    prior=dict(np.load(d/"train_only_head.npz"))
    evidence=dict(np.load(temporal/"train_only_evidence.npz"))
    rooms=load(temporal/"assets/protocol.json")["rooms"]
    actor_paths=sorted((base/"test").glob("*.actor.json"))
    if len(actor_paths)!=12*len(rooms):raise RuntimeError("Incomplete event cohort")
    episodes={};scans={};requests={};sidpaths={}
    for file in actor_paths:
        sid=file.name.split(".actor")[0];a=load(file);sidpaths[sid]=file
        hist=a["history"];xyz,heading=hist[0][2],hist[0][3]
        end,look=endpoints(a["actions"],np.array(xyz),heading)
        rows={}
        for t,_,pos,face,_,key in hist:
            if t%4:continue
            dist=np.linalg.norm(end-pos,axis=1)
            sides=np.where((dist<=.75)&((look@face)>=math.cos(math.pi/6+.001)))[0]
            # Eligibility to update comes solely from executed odometry and current camera heading.
            if len(sides)==1:
                rows[t]=(int(sides[0]),key)
                requests["verify:"+key]=base/"rgb"/(key+".png")
        scans[sid]=rows
    encoder.embed(requests)
    for sid,path in sidpaths.items():
        a=load(path); hist=a["history"];table=scans[sid]
        x=prior_feature(encoder,base/"rgb"/(hist[0][-1]+".png"))
        ps=.18*np.tanh(margins(prior,x)/evidence["scale"][:,None,None])
        seen=np.full(2,-1);memory=np.zeros((2,2));permanent=np.zeros((2,2));has_persist=np.zeros(2,bool)
        check=[v["t"] for v in a["checkpoints"]];result=[]
        for row in hist:
            t=row[0]
            if t in table:
                side,key=table[t];image=encoder.cache["verify:"+key]
                value=unit(image-evidence["center"])@evidence["xw"].T@evidence["alpha"]
                # Strong positive support for currently visible target; clear, adequately
                # re-inspected locations can downweight previously positive evidence.
                signed=np.clip(value[:2]-np.maximum(value[:2][::-1],value[2]),-1.,1.)
                memory[side]=signed if seen[side]<0 else .15*memory[side]+.85*signed
                seen[side]=t
                permanent[side]=signed if not has_persist[side] else np.maximum(permanent[side],signed)
                has_persist[side]=True
            if t not in check:continue
            fresh=memory*np.where(seen>=0,np.exp2(-(t-seen)/96.),0.)[:,None]
            by_arm=dict(jepa_prior_only=ps[0],direct_prior_only=ps[1],
                        jepa_plus_update=ps[0]+fresh,direct_plus_update=ps[1]+fresh,
                        uniform_plus_update=fresh,jepa_permanent_positive=ps[0]+permanent)
            for _ in range(check.count(t)):
                result.append(dict(t=t,scores={k:v.tolist() for k,v in by_arm.items()},
                                   last_inspection=seen.tolist(),recent_evidence=fresh.tolist()))
        episodes[sid]=result
    (temporal/"blind_predictions.json").write_text(json.dumps(episodes,allow_nan=False))
    print("SCORED",len(episodes),len(requests),flush=True)


def evaluate(d):
    temporal=d/"temporal";base=temporal/"streams";episodes=load(temporal/"blind_predictions.json")
    outcomes={};couples={};paired=0;first=[];maxdiff=0.
    for sid,rows in episodes.items():
        t=load(base/"test"/(sid+".truth.json"));hist=load(base/"test"/(sid+".actor.json"))["history"]
        counter=sid[:-1]+str(1-int(t["relocated"]))
        if not t["relocated"]:
            other=load(base/"test"/(counter+".actor.json"))["history"]
            diff=[r[0] for r,s in zip(hist,other) if r[-1]!=s[-1]]
            assert diff and min(diff)>180
            first.append(min(diff));paired+=1
            for r,s in zip(rows[:5],episodes[counter][:5]):
                for arm in r["scores"]:
                    maxdiff=max(maxdiff,np.max(np.abs(np.array(r["scores"][arm])-s["scores"][arm])))
        layout="moved" if t["relocated"] else "stationary"
        for i,(r,gt) in enumerate(zip(rows,t["labels"])):
            target=np.array(gt["query_routes"]).T;assert r["t"]==gt["t"]
            for arm,scores in r["scores"].items():
                delta=(np.array(scores)*(target*2-1)).sum(axis=0)
                hit=(delta>1e-7).astype(float)+.5*(np.abs(delta)<=1e-7)
                for group in ("all",t["cue"],t["house"]):
                    key="/".join((layout,gt["phase"],group))
                    outcomes.setdefault(key,{}).setdefault(arm,[]).extend(hit)
                if i in (2,6):
                    couples.setdefault((sid,arm),[]).append(hit)
    assert maxdiff<1e-8
    means={k:{arm:float(np.mean(v)) for arm,v in a.items()} for k,a in outcomes.items()}
    improvements={}
    for (sid,arm),(before,after) in couples.items():
        moved=sid.endswith(".1")
        group="correct_old_recovered_new" if moved else "correct_static_preserved"
        for initial,now in zip(before,after):
            if initial==1:improvements.setdefault(group,{}).setdefault(arm,[]).append(float(now==1))
    cohort=dict(cases=len(episodes),queries_per_case=2,physical_relocations=paired,matched_events=paired,
        first_differentiating_rgb_minmax=[min(first),max(first)],predictions_before_new_observation_max_delta=float(maxdiff),
        train_houses=8,new_dynamic_rooms=6,categories=["cup","sofa"],actual_actions=300,
        phases_at_steps=[(r["phase"],r["t"]) for r in t["labels"]],
        per_probe="24 outward + 36 return/reorientation = 60 actions per visit",
        prior_weights_sha256=sha(d/"train_only_head.npz"),evidence_weights_sha256=sha(temporal/"train_only_evidence.npz"),
        model_sha256=sha(Path(__file__)),protocol_sha256=sha(temporal/"assets/protocol.json"),
        conditional={k:{a:[int(sum(v)),len(v)] for a,v in arms.items()} for k,arms in improvements.items()},
        accuracy=means,limitations="Two object categories, mutually exclusive occupant in a fork, scripted history, target-locating task, known nominal routes. Not multi-instance search SR.")
    (temporal/"summary.json").write_text(json.dumps(cohort,indent=2))
    print(json.dumps({k:v for k,v in means.items() if k.endswith("/all")},indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root",type=Path);ap.add_argument("experiment",type=Path)
    ap.add_argument("step",choices=("train","score","evaluate"))
    a=ap.parse_args();d=a.experiment.resolve()
    if a.step=="evaluate":evaluate(d);return
    root=a.root.resolve();sys.path.insert(0,str(root/"jepa_five_20261123/clip_v5"))
    from clip_views import Clip
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    encoder=Encoder(Clip())
    if a.step=="train":train(d,encoder)
    else:score(d,encoder)


if __name__=="__main__":main()
