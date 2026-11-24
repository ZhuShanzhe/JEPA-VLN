"""Paired action-conditioned future-feature and direct-supervision route heads."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEEDS=(11,29,47)
UPDATES=480
BATCH=48
TAPS=(.25,.5,.75,1.)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def route_inputs(ctx,actions):
    """Use nominal actions only. Index features by walked direction and departure observations."""
    yaw=np.cumsum((actions==1).astype(float)-(actions==2),axis=1)*math.pi/6
    move=(actions==0)*.25
    dx=np.cumsum(move*np.cos(yaw),axis=1)
    dy=np.cumsum(move*np.sin(yaw),axis=1)
    steps=np.asarray([round(f*actions.shape[1])-1 for f in TAPS])
    x,y,face=dx[:,steps],dy[:,steps],yaw[:,steps]
    bearing=np.where(np.hypot(x,y)>.05,np.arctan2(y,x),face)
    # Three observed headings, each with full-frame, left region and right region.
    heads=np.deg2rad(np.array([-30,0,30])[:,None]+np.array([0,22.5,-22.5])[None,:]).ravel()
    cos=np.cos(bearing[:,:,None]-heads)
    att=np.exp(5.*(cos-cos.max(-1,keepdims=True)));att/=att.sum(-1,keepdims=True)
    anchor=np.einsum('ctv,vd->ctd',att,ctx.reshape(9,-1))
    scene=np.broadcast_to(ctx.mean(axis=(0,1)),anchor.shape)
    endcos=np.cos(face[:,:,None]-heads)
    ang=np.argmax(endcos,axis=-1)
    look=ctx.reshape(9,-1)[ang]
    geo=np.stack([x/6.,y/6.,np.sin(face),np.cos(face),
                  np.hypot(x,y)/6.,np.broadcast_to(steps+1,x.shape)/28.,
                  cos.max(-1),endcos.max(-1)],-1)
    return np.concatenate([anchor,look,scene,geo],axis=-1).astype('float32')


class Ranker(nn.Module):
    def __init__(self,dim):
        super().__init__();self.dim=dim
        self.encoder=nn.Sequential(nn.Linear(3*dim+8,144),nn.LayerNorm(144),nn.GELU(),nn.Linear(144,128),nn.GELU())
        self.project=nn.Linear(128,3*dim)
        self.goal=nn.Linear(dim,32)
        self.readout=nn.Sequential(nn.Linear(128+32+3+8,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,x,txt):
        h=self.encoder(x)
        visible=F.normalize(x[...,:self.dim],dim=-1)
        pred=F.normalize(visible.unsqueeze(-2)+self.project(h).reshape(*h.shape[:-1],3,self.dim),dim=-1)
        match=(pred*txt[:,None,None,None,:]).sum(-1)
        goal=self.goal(txt)[:,None,None,:].expand(*h.shape[:-1],32)
        logits=self.readout(torch.cat([h,goal,8*match,x[...,-8:]],dim=-1)).squeeze(-1)
        return torch.logsumexp(logits,dim=-1)-math.log(logits.shape[-1]),pred


def remem(root):
    folder=root/'jepa_memory_v10_20261124/models';sys.path.insert(0,str(folder))
    from v10_live import RouteMemory
    return RouteMemory(folder/'v10.json')


def memory_start(mem,cue):
    mem.reset()
    for t,j in enumerate((2,0,1)):
        a=math.radians((-30,0,30)[j])
        mem.observe([0,0],t,[math.cos(a),math.sin(a)],float(cue[j]))


def bonus(mem,actions,step):
    v=mem.features(actions,[0,0],[1,0],step)
    return v@np.array([mem.weights[k] for k in ('novelty','clearance','target_trace')])


def read(root,split,sources,teacher=True):
    exp=root/'jepa_hidden_v11/models/cache'; out=[]; mem=remem(root)
    for src in sources:
        folder=exp/src/split
        for file in sorted((folder/'actor').glob('*.npz')):
            with np.load(file) as z:
                name=str(z['house']);orig=str(z['origin']);words=z['words'].tolist()
                acts=z['actions'];x=route_inputs(z['ctx'].astype('float32'),acts)
                text=z['text'];cue=z['cue']
            with np.load(folder/'truth'/file.name) as z:
                labels=z['y'];groups=z['group'].tolist()
                extra={k:z[k] for k in z.files if k not in ('y','group')}
            future=None
            if teacher:
                with np.load(folder/'teacher'/file.name) as z:future=z['future'].astype('float32')
            for q,w in enumerate(words):
                memory_start(mem,cue[:,q])
                out.append(dict(x=x,text=text[q],future=future,y=labels[:,q].astype('float32'),
                                house=name,source=src,origin=orig,group=groups[q],word=w,q=q,
                                actions=acts,key=file.stem,cue=cue[:,q],initial=bonus(mem,acts,2),
                                row=file.name,extra=extra))
    return out


def sample(batch):
    def stack(k):return torch.tensor(np.stack([x[k] for x in batch]),device='cuda')
    return stack('x'),stack('text'),stack('future'),stack('y')


def training(root,sources,rows,seed,arm,dest):
    rng=np.random.default_rng(seed)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=Ranker(len(rows[0]['text'])).cuda()
    optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    table=defaultdict(lambda: defaultdict(list))
    for i,r in enumerate(rows):table[r['source']][r['house']].append(i)
    sources=sorted(table);houses={s:sorted(table[s]) for s in sources}
    meter=[]
    for step in range(UPDATES):
        batch=[]
        for _ in range(BATCH):
            src=sources[int(rng.integers(len(sources)))];hs=houses[src][int(rng.integers(len(houses[src])))]
            ids=table[src][hs];batch.append(rows[ids[int(rng.integers(len(ids)))]])
        x,txt,future,y=sample(batch);score,pred=model(x,txt)
        target=y.gt(0).float()
        bce=F.binary_cross_entropy_with_logits(score,target)
        mask=y[:,:,None]>y[:,None,:];pair=F.softplus(score[:,None,:]-score[:,:,None])
        rank=(pair*mask).sum()/mask.sum().clamp_min(1)
        # Match actual future embedding and distinguish physical alternatives within origin.
        if arm=='wrong_future':future=future.roll(shifts=1,dims=1)
        direct=1.-(pred*future).sum(-1).mean()
        dp=F.normalize(pred-pred.mean(1,keepdim=True),dim=-1)
        dt=F.normalize(future-future.mean(1,keepdim=True),dim=-1)
        latent=direct+.5*(1.-(dp*dt).sum(-1).mean())
        loss=bce+rank+(.65*latent if arm!='direct' else 0.)
        optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);optim.step()
        if step in (0,UPDATES-1):
            meter.append([step,float(bce.detach()),float(rank.detach()),float(latent.detach())])
    path=dest/f'{arm}_{seed}.pt'
    torch.save(model.cpu().state_dict(),path)
    return dict(arm=arm,seed=seed,params=sum(p.numel() for p in model.parameters()),
                checkpoint_sha256=sha(path),objective_samples=meter)


def write(path,obj):
    path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(obj,indent=2,allow_nan=False))


def fit(root):
    d=root/'jepa_hidden_v11/models'
    gc=root/'jepa_memory_v10_20261124/models/v10.json'
    cfg=dict(protocol='v11-paired-fixed',seeds=SEEDS,steps=UPDATES,batch=BATCH,
             learning_rate=.001,latent_weight=.65,differential_latent_weight=.5,
             memory_weights=sha(gc),fit_sources=['hssd','combined'],horizon_taps=TAPS,
             starting_headings_degrees=[-30,0,30],primary='new physical goal-instance opportunity on first choice',
             allowed_test_inputs='three observed color views, relative headings, frozen target text, equal-length planned actions, same episode-local memory',
             declared_before_blind_test=True,source_sha256=sha(Path(__file__)))
    cfg_file=d/'protocol.json'
    if cfg_file.exists():assert json.loads(cfg_file.read_bytes())==json.loads(json.dumps(cfg))
    else:write(cfg_file,cfg)
    rows=read(root,'train',('hssd','blender'))
    manifests={}
    for mode in ['hssd','combined']:
        rr=[r for r in rows if mode=='combined' or r['source']=='hssd']
        assert rr and (mode=='hssd' or len({r['source'] for r in rr})==2)
        dest=d/'weights'/mode;dest.mkdir(exist_ok=True,parents=True)
        manifest=dest/'fit.json'
        if manifest.exists():manifests[mode]=json.loads(manifest.read_bytes());continue
        ss=[training(root,None,rr,seed,arm,dest) for seed in SEEDS for arm in ('direct','jepa','wrong_future')]
        manifests[mode]=dict(n_query_origins=len(rr),n_origins=len({(r['source'],r['house'],r['origin']) for r in rr}),
                             houses=sorted({(r['source'],r['house']) for r in rr}),runs=ss)
        write(manifest,manifests[mode]);print('fit',mode,len(rr),flush=True)
    write(d/'fit_receipt.json',manifests)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('project_root',type=Path)
    fit(ap.parse_args().project_root)
