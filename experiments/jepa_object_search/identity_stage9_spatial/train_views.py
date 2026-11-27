"""Fit an object-relative future-view predictor on observations of the same HSSD instance."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from view_jepa import ViewPredictor, camera_features, crop_features

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SEED=290713


def inputs():
    base=ROOT/"jepa_reid_v20"
    sys.path.insert(0,str(base/"model"))
    from reid_model import Encoder, mask_context, preprocess
    from train import Crops
    seed_root=ROOT/"identity_stage3_model/train/private"
    sets=[(base/"data/manifest.jsonl",base/"data",base/"model/features.pt"),
          (base/"clocks/manifest.jsonl",base/"clocks",base/"clocks/evaluation/features.pt")]
    for s in ("train","val"):
        sets.append((seed_root/(s+"_clock_seeds.jsonl"),seed_root/(s+"_clock_anchor"),None))
    records=[]; source=[]; codes=[]; hashes=[]
    encoder=None
    for manifest, path, checkpoint in sets:
        rows=[json.loads(line) for line in manifest.read_bytes().splitlines()]
        sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
        hashes.append(sha)
        if checkpoint:
            ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
            assert ck["manifest_sha256"]==sha and ck["representation"].startswith("dinov2_vitb14")
            visual=ck["features"]
        else:
            if encoder is None:
                encoder=Encoder(ROOT).to("cuda").eval()
            crops=Crops(path,rows)
            enc=[]
            for start in range(0,len(rows),12):
                pair=[crops[i] for i in range(start,min(start+12,len(rows)))]
                a,b=map(torch.stack,zip(*pair))
                enc.append(encoder(a.cuda(),b.cuda()).cpu())
            visual=torch.cat(enc)
        assert len(rows)==len(visual)
        codes.append(F.normalize(visual[:,:1536].float()).numpy())
        for r in rows:
            with Image.open(path/r["context_image"]) as im:
                crop=np.array(im.convert("RGB").crop(r["box_in_context"]))
            source.append(crop_features(crop,r["bbox"],(384,384)))
            records.append(r)
    return records,np.stack(source),np.concatenate(codes),hashes


def assemble(rows,src,enc):
    byinst=defaultdict(list); byroom=defaultdict(list); families=defaultdict(set)
    for i,r in enumerate(rows):
        byinst[(r["house"],r["instance"])].append(i)
        byroom[(r["house"],r["category"])].append(i)
        families[r["split"]].add(r["house"].split("_")[0])
    assert all(not families[a]&families[b] for a,b in [('train','test'),('train','val'),('val','test')])
    train_ids=np.array([i for i,r in enumerate(rows) if r["split"]=="train"])
    # Fixed normalized DINO target projected onto train-only principal directions.
    def project(features,dims):
        mu=features[train_ids].mean(0); x=features-mu
        _,_,v=torch.pca_lowrank(torch.from_numpy(x[train_ids]),q=dims+8,center=False,niter=3)
        a=x@v[:,:dims].numpy()
        return np.clip(a/np.maximum(a[train_ids].std(0),1e-4),-6,6)
    z=np.column_stack([project(enc,32),project(src[:,96:192],16),
                       np.clip((src[:,192:195]-src[train_ids,192:195].mean(0))/
                               np.maximum(src[train_ids,192:195].std(0),.03),-6,6)]).astype("f")
    dist_margin={}
    for members in byroom.values():
        intra=enc[members]@enc[members].T
        ids=np.array([rows[i]['instance'] for i in members])
        for k,i in enumerate(members):
            peers=intra[k,ids!=ids[k]]
            # No impostor in the scene is uninformative for discriminability.
            dist_margin[i]=np.clip((1-float(peers.max()))/.5,0,1) if len(peers) else .5
    result=defaultdict(list)
    for _,members in byinst.items():
        if len(members)<2: continue
        frames=[rows[i] for i in members]
        eyes=np.array([r["camera_xyz"] for r in frames])[:,[0,2]]
        angle=np.radians([r["yaw_deg"] for r in frames])
        normals=np.column_stack([-np.sin(angle),np.cos(angle)])
        if np.linalg.cond(normals)>25:continue
        center=np.linalg.lstsq(normals,(eyes*normals).sum(1),rcond=None)[0]
        radii=np.linalg.norm(eyes-center,axis=1)
        if min(radii)<.7 or max(radii)>4.2 or np.max(np.abs(((eyes-center)*normals).sum(1)))>.03:continue
        projected=np.array([r["pixels"] for r in frames])*radii**2
        for a,i in enumerate(members):
            for b,j in enumerate(members):
                if i==j or np.linalg.norm(eyes[a]-eyes[b])>3.0:continue
                r=rows[j]; x,y,u,v=r["bbox"]
                if abs(r["pitch_deg"]-rows[i]["pitch_deg"])>20:continue
                angle_delta=(angle[b]-angle[a]+np.pi)%(2*np.pi)-np.pi
                ratio=radii[b]/radii[a]
                legible=np.clip(min(u-x,v-y)/65.,0,1)
                framed=np.clip(min(x,y,384-u,384-v)/15.,0,1)
                face=np.clip(projected[b]/projected.max(),0,1)
                utility=.35*legible+.20*framed+.30*face+.15*dist_margin[j]
                result[r["split"]].append((i,j,angle_delta,ratio,utility))
    return result, z


def arrays(sequence,features,z,rows):
    a=np.array(sequence)
    i,j=a[:,:2].T.astype(int)
    return dict(x=np.column_stack([features[i],camera_features(a[:,2],a[:,3])]).astype('f'),y=a[:,4].astype('f'),
                z=z[j],anchor=i,house=np.array([rows[k]['house'] for k in i]),clock=np.array([rows[k]['category']=='clock' for k in i]))


def evaluate(d,q,mask=None):
    group=defaultdict(list)
    for idx in np.flatnonzero(mask if mask is not None else np.ones(len(q),bool)):
        group[d['anchor'][idx]].append(idx)
    chosen=[];random=[];best=[];hits=[]
    for ids in group.values():
        if len(ids)<2:continue
        y=d['y'][ids]; pred=q[ids]; val=y[np.isclose(pred,pred.max(),atol=1e-7,rtol=0)].mean()
        chosen.append(val);random.append(y.mean());best.append(y.max());hits.append(np.isclose(val,y.max(),atol=1e-6))
    if not chosen:return {'choices':0}
    return dict(choices=len(chosen),selected=float(np.mean(chosen)),random=float(np.mean(random)),
                oracle=float(np.mean(best)),best_view_rate=float(np.mean(hits)))


def train():
    torch.set_num_threads(4); torch.manual_seed(SEED)
    rows,features,embedding,sha=inputs()
    splits,z=assemble(rows,features,embedding)
    ds={k:arrays(a,features,z,rows) for k,a in splits.items()}
    train,val,test=(ds[k] for k in ('train','val','test'))
    X=train['x']; mean=X.mean(0); std=X.std(0).clip(.025)
    norm=lambda x:torch.from_numpy(((x-mean)/std).clip(-8,8))
    xt=norm(X); xv=norm(val['x']); yt=torch.from_numpy(train['y']); zt=torch.from_numpy(train['z'])
    siblings=defaultdict(list)
    for i,g in enumerate(train['anchor']): siblings[g].append(i)
    partner=np.arange(len(X)); rng=np.random.default_rng(SEED)
    for ids in siblings.values():
        order=rng.permutation(ids);partner[order]=np.roll(order,1)
    partner=torch.from_numpy(partner)
    out=HERE/"weights";out.mkdir(exist_ok=True)
    result=dict(seed=SEED,manifest_hashes=sha,unique_views=len(rows),future_latent="32 train-PCA DINOv2 crop, 16 gradient-grid and 3 shape/size dimensions",
                utility="0.35 legibility + 0.20 framing + 0.30 normalized projected surface + 0.15 category distinctiveness",
                splits={},scores={})
    for k,d in ds.items():
        result["splits"][k]=dict(houses=len(set(d['house'])),transitions=len(d['x']),clock_transitions=int(d['clock'].sum()),
            instances=len({rows[i]['instance'] for i in d['anchor']}),categories=len({rows[i]['category'] for i in d['anchor']}))
    for name,aux in [('direct',False),('view_jepa',True)]:
        torch.manual_seed(SEED); gen=torch.Generator().manual_seed(SEED)
        model=nn.Sequential(nn.Linear(X.shape[1],128),nn.ReLU(),nn.Linear(128,z.shape[1]),nn.Tanh(),nn.Linear(z.shape[1],1),nn.Sigmoid())
        opt=torch.optim.AdamW(model.parameters(),lr=.0015)
        best=-float('inf')
        for epoch in range(36):
            model.train()
            for k in torch.randperm(len(X),generator=gen).split(384):
                latent=model[:3](xt[k]); pred=model[3:](latent).flatten()
                loss=F.mse_loss(pred,yt[k])
                if aux:
                    v=partner[k]; other=model[:3](xt[v])
                    loss=loss+.075*F.smooth_l1_loss(latent,zt[k])+.12*F.smooth_l1_loss(latent-other,zt[k]-zt[v])
                opt.zero_grad();loss.backward();opt.step()
            model.eval()
            with torch.inference_mode(): score=model(xv).numpy().ravel()
            metric=evaluate(val,score)['selected']
            if metric>best:
                best=metric;best_epoch=epoch+1
                np.savez(out/(name+".npz"),mean=mean,std=std,**{k:p.detach().numpy() for k,p in zip(('w1','b1','w2','b2','w3','b3'),model.parameters())})
        network=ViewPredictor(out/(name+'.npz'))
        q,latent=network.forward(test['x'])
        stats=dict(epoch=best_epoch,validation_selected=best,overall=evaluate(test,q),clocks=evaluate(test,q,test['clock']))
        if aux:
            # Permute only intended camera moves within each identical source; no target is fed to the predictor.
            shuffled=test['x'].copy()
            groups=defaultdict(list)
            for idx,key in enumerate(test['anchor']):groups[key].append(idx)
            for ids in groups.values():shuffled[ids,-7:]=test['x'][np.roll(ids,1),-7:]
            null,lc=network.forward(shuffled)
            stats['angle_range_shuffle']=evaluate(test,null)
            for k,sl in [('dino',slice(0,32)),('gradient_shape',slice(32,None))]:
                stats[k+'_future_mse']=float(np.mean((latent[:,sl]-test['z'][:,sl])**2))
                stats[k+'_shuffled_mse']=float(np.mean((lc[:,sl]-test['z'][:,sl])**2))
        result['scores'][name]=stats
    result['deployed_sha256']=hashlib.sha256((out/'view_jepa.npz').read_bytes()).hexdigest()
    (HERE/'view_training.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['scores']),flush=True)


if __name__=='__main__':
    train()
