"""Exploratory, long-path-only grouped residual. The released joint camera predictor is immutable."""
import hashlib
import json
from pathlib import Path
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_joint import ROOT, JointPredictor, load, scored, metrics, file_hash

SEED=1128
EPOCHS=36


class BranchResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.source=nn.Linear(1024,96)
        self.path=nn.Sequential(nn.Linear(32*3+7+1,128),nn.LayerNorm(128),nn.GELU(),nn.Linear(128,96))
        self.mapping=nn.Sequential(nn.Linear(96*3,128),nn.LayerNorm(128),nn.GELU(),nn.Linear(128,1024))

    def forward(self,source,sequence,pose,n,base):
        # Ordered, native primitive counts in consecutive 4-step windows (128-step padding).
        code=F.one_hot(sequence.long(),4)[...,1:].float().reshape(len(sequence),32,4,3).mean(2)
        p=self.path(torch.cat((code.flatten(1),pose,n[:,None]/108),1))
        s=self.source(F.normalize(source,dim=-1)*32)
        delta=self.mapping(torch.cat((p,s,p*s),1))
        return F.normalize(base+.02*delta,dim=-1)


def grouped(d,split):
    bins={}
    for idx in np.flatnonzero(d["split"]==split):
        key=(d["house"][idx],d["group"][idx],d["horizon"][idx])
        bins.setdefault(key,[]).append(idx)
    groups=[]
    for g in bins.values():
        assert len(g)==3,"the prespecified action trees must have three branches"
        norm=F.normalize(torch.from_numpy(d["future"][g]),dim=-1).numpy()
        if np.min(1-norm@norm.T+2*np.eye(3))>=1e-6:groups.append(g)
    return np.asarray(groups,dtype=np.int64)


def retrieve(pred,g,actual):
    def acc(a,b):
        sims=np.einsum("gix,gjx->gij",a/np.maximum(np.linalg.norm(a,axis=2,keepdims=True),1e-8),
                       b/np.maximum(np.linalg.norm(b,axis=2,keepdims=True),1e-8))
        winners=np.isclose(sims,sims.max(2,keepdims=True),rtol=0,atol=1e-7)
        return float((np.diagonal(winners,axis1=1,axis2=2)/np.maximum(1,winners.sum(2))).mean())
    a,b=pred[g],actual[g]
    return dict(groups=len(g),raw=acc(a,b),centered=acc(a-a.mean(1,keepdims=True),b-b.mean(1,keepdims=True)))


def main():
    os.umask(0o077)
    np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(3)
    path=ROOT/"private_train"
    checkpoint=path/"joint_long.pt"
    fingerprint=file_hash(checkpoint)
    device="cuda" if torch.cuda.is_available() else "cpu"
    d=load("long")
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
    frozen=JointPredictor(saved["dims"]).to(device).eval()
    frozen.load_state_dict(saved["weights"])
    stats=saved["stats"]["long"].numpy()
    base=scored(frozen,"long",d,np.arange(len(d["context"])),stats,"joint_long",device)[0]
    residual=BranchResidual().to(device)
    optim=torch.optim.AdamW(residual.parameters(),lr=.0005,weight_decay=.01)
    tensors=[torch.as_tensor(x,device=device) for x in (d["context"][:,:1024],d["action"].astype(np.uint8),
             np.clip((d["pose"]-stats[0])/stats[1],-8,8),d["horizon"].astype(np.float32),base)]
    teacher=torch.as_tensor(d["future"],device=device)
    get_pred=lambda ids:residual(*(t[ids] for t in tensors))
    train=grouped(d,"train"); val=grouped(d,"val")
    rng=np.random.default_rng(SEED)
    best=-1
    for epoch in range(1,EPOCHS+1):
        residual.train()
        for chunk in np.array_split(rng.permutation(len(train)),int(np.ceil(len(train)/40))):
            ids=train[chunk]
            pred=get_pred(ids.flatten()).view(-1,3,1024)
            actual=teacher[ids]
            a=F.normalize(pred-pred.mean(1,keepdim=True),dim=-1)
            b=F.normalize(actual-actual.mean(1,keepdim=True),dim=-1)
            sims=a@b.transpose(1,2)/.15
            labels=torch.arange(3,device=device).repeat(len(ids))
            contrast=(F.cross_entropy(sims.reshape(-1,3),labels)+F.cross_entropy(sims.transpose(1,2).reshape(-1,3),labels))*.5
            pointwise=(1-(pred*actual).sum(-1)).mean()
            loss=pointwise+.15*contrast
            optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(residual.parameters(),2.);optim.step()
        with torch.inference_mode():
            residual.eval()
            ids=np.unique(val)
            pred=np.zeros_like(base)
            for chunk in np.array_split(ids,int(np.ceil(len(ids)/256))):
                pred[chunk]=get_pred(chunk).cpu().numpy()
            # Calibration selects on ordinary raw branch matching of the three longest horizons.
            score=float(np.mean([retrieve(pred,val[d["horizon"][val[:,0]]==h],d["future"])["raw"] for h in (52,80,108)]))
            if score>best:
                best=score;epoch_best=epoch
                torch.save(dict(weights=residual.state_dict(),base_sha256=fingerprint,epoch=epoch),path/"long_branch_residual.pt")
    residual.load_state_dict(torch.load(path/"long_branch_residual.pt",map_location=device,weights_only=True)["weights"])
    residual.eval()
    test=grouped(d,"test")
    pred=np.zeros_like(base)
    with torch.inference_mode():
        ids=np.flatnonzero(d["split"]=="test")
        for chunk in np.array_split(ids,int(np.ceil(len(ids)/256))):
            pred[chunk]=get_pred(chunk).cpu().numpy()
    report=dict(phase="Exploratory secondary experiment after first test readout; shared held-out panel.",seed=SEED,
                training_groups=len(train),val_groups=len(val),test_groups=len(test),
                epochs=EPOCHS,chosen_epoch=epoch_best,val_long_macro_raw_accuracy=best,base_sha256=fingerprint,
                pose_path="camera inference uses untouched joint_long checkpoint, no residual applied",
                results={})
    for h in (12,24,52,80,108):
        g=test[d["horizon"][test[:,0]]==h]
        m=(d["split"]=="test")&(d["horizon"]==h)
        report["results"][str(h)]={k:{**retrieve(p,g,d["future"]),
                    "future_cosine":float((p[m]*F.normalize(torch.from_numpy(d["future"][m]),dim=1).numpy()).sum(1).mean())}
                    for k,p in (("base",base),("residual",pred))}
    assert fingerprint==file_hash(checkpoint)
    (ROOT/"branch_residual_metrics.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__": main()
