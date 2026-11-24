"""Calibrated, bounded object-conditioned residual on a frozen V11 route predictor."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'models'))
from train_routes import SEEDS, digest, load_prior, measure, pairs, previous, write


CAPS = (0., .03, .06, .10, .15)
STEPS = (160,320,480)
ARMS = ('rv_direct','rv_jepa')


class ForegroundResidual(nn.Module):
    def __init__(self, dim, mean_text):
        super().__init__()
        self.dim=dim
        self.register_buffer('mean',mean_text)
        self.encode=nn.Sequential(nn.Linear(2*dim+8,96),nn.LayerNorm(96),nn.GELU(),nn.Linear(96,2*dim))
        self.text_proj=nn.Linear(dim,dim,bias=False)
        nn.init.zeros_(self.text_proj.weight)

    def forward(self,x,pred,text):
        # V11 predictions and observed look-direction patch, both available before motion.
        start=x[...,:self.dim]
        latent=pred.mean(dim=-2)
        hidden=self.encode(torch.cat((start,latent,x[...,-8:]),-1))
        slots=F.normalize(latent.unsqueeze(-2)+hidden.reshape(*latent.shape[:-1],2,self.dim),dim=-1)
        query=text-self.mean
        query=F.normalize(query+.25*self.text_proj(query),dim=-1)
        similarity=12*(slots*query[:,None,None,None,:]).sum(-1)
        score=torch.logsumexp(similarity.flatten(-2),dim=-1)
        # Zero common component. The additive route probability correction is bounded by cap.
        correction=torch.tanh(score-score.mean(-1,keepdim=True))
        return correction,slots


def read(project,split,labels=False,local=False):
    root=project/'jepa_route_v12'
    base=project/'jepa_hidden_v11' if split in ('train','calibration') else root
    cache=base/'models'/'cache'/'hssd'/split
    old=previous(project)
    mem=old.remem(project)
    keys=[]
    values=defaultdict(list)
    for path in sorted((cache/'actor').glob('*.npz')):
        with np.load(path) as f:
            house,orig,words=str(f['house']),str(f['origin']),f['words'].tolist()
            plans,cues=f['actions'],f['cue']
            x=old.route_inputs(f['ctx'].astype('float32'),plans)
            text=f['text'].astype('float32')
        if labels:
            with np.load(cache/'truth'/path.name) as f:
                y=f['y'].astype('float32')
        if local:
            with np.load(root/'data'/'local_targets'/split/f'{orig}.npz') as f:
                assert words==f['words'].tolist() and np.array_equal(f['y'],y)
                z=f['local_z'].astype('float32')
                ts=f['first_tap'].astype('int64')
                assert np.array_equal(ts>=0,y>0)
        for i,word in enumerate(words):
            keys.append((house,orig,word))
            old.memory_start(mem,cues[:,i])
            values['memory'].append(old.bonus(mem,plans,2).astype('float32'))
            values['x'].append(x)
            values['text'].append(text[i])
            if labels:
                values['y'].append(y[:,i])
            if local:
                values['features'].append(z[:,i])
                values['first'].append(ts[:,i])
    assert keys and len(set(keys))==len(keys)
    return keys,{k:torch.as_tensor(np.stack(v),device='cuda') for k,v in values.items()}


@torch.inference_mode()
def prior_features(prior,data):
    scores,forecasts=[],[]
    for a in range(0,len(data['text']),96):
        s,p=prior(data['x'][a:a+96],data['text'][a:a+96])
        scores.append(s.sigmoid())
        forecasts.append(p)
    return torch.cat(scores),torch.cat(forecasts)


def train(seed,arm,tr,keys,conflicts,prior,pred,mean,out):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed)
    model=ForegroundResidual(tr['text'].shape[-1],mean).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.025)
    buckets=defaultdict(list)
    for j,k in enumerate(keys):
        buckets[k[0]].append(j)
    homes=sorted(buckets)
    for step in range(1,max(STEPS)+1):
        ix=[int(rng.choice(buckets[homes[j]])) for j in rng.integers(len(homes),size=32)]
        opposite=[conflicts[j] for j in rng.integers(len(conflicts),size=8)]
        ix += [a for a,b in opposite]+[b for a,b in opposite]
        residual,slots=model(tr['x'][ix],pred[ix],tr['text'][ix])
        raw=(prior[ix]+.15*residual).clamp(.005,.995)
        logits=torch.logit(raw)
        seen=tr['y'][ix]>0
        rank=(tr['y'][ix].unsqueeze(-1)>tr['y'][ix].unsqueeze(-2))
        difference=logits.unsqueeze(-1)-logits.unsqueeze(-2)
        loss=F.binary_cross_entropy(raw,seen.float())+F.softplus(-difference)[rank].mean()+.025*residual.square().mean()
        # Shared initial state, different requested instances. Anchor the global task with a lower swap weight.
        rs=logits[32:40]-logits[40:]
        ab=seen[32:40].unsqueeze(-1)&seen[40:].unsqueeze(-2)
        loss=loss+.18*F.softplus(.4-rs.unsqueeze(-1)+rs.unsqueeze(-2))[ab].mean()
        if arm=='rv_jepa':
            b,r=torch.where(seen)
            when=tr['first'][ix][b,r]
            truth=tr['features'][ix][b,r]
            # Attend to the teacher instance only in training. Two slots allow competing foreground objects.
            matched=(slots[b,r,when]*truth[:,None,:]).sum(-1).max(dim=-1).values
            other=(slots[b,:,when]*truth[:,None,None,:]).sum(-1).max(dim=-1).values
            n=~seen[b]
            loss=loss+.55*(1-matched).mean()+.25*F.relu(.12+other-matched[:,None])[n].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        opt.step()
        if step in STEPS:
            torch.save(model.state_dict(),out/f'{arm}_{seed}_{step}.pt')
    return model


def balanced(keys,truth,picks):
    buckets=defaultdict(list)
    for k,y,j in zip(keys,truth,picks):
        buckets[k[:2]].append(float(y[j]>0))
    houses=defaultdict(list)
    for (house,_),vals in buckets.items():
        houses[house].append(np.mean(vals))
    return np.array([np.mean(houses[h]) for h in sorted(houses)])


@torch.no_grad()
def predict(model,d,fore):
    out=[]
    for i in range(0,len(fore),96):
        out.append(model(d['x'][i:i+96],fore[i:i+96],d['text'][i:i+96])[0])
    return torch.cat(out)


def fit(project):
    out=project/'jepa_route_v12/refinement'
    lock=out/'selection.json'
    if lock.exists():raise FileExistsError(lock)
    folder=out/'weights'
    folder.mkdir(exist_ok=True)
    keys,tr=read(project,'train',True,True)
    extra,more=read(project,'extra_train',True,True)
    if len({k[0] for k in extra})!=12:
        raise ValueError('extra training split not complete')
    keys+=extra
    tr={k:torch.cat((v,more[k])) for k,v in tr.items()}
    ck,ca=read(project,'calibration',True)
    assert len(set(keys))==len(keys) and not ({q[0] for q in ck}&{q[0] for q in keys})
    y=ca['y'].cpu().numpy()
    cp=pairs(ck,y);tp=pairs(keys,tr['y'].cpu().numpy())
    word_indices={w:i for i,(_,_,w) in enumerate(keys)}
    mean=tr['text'][list(word_indices.values())].mean(0)
    report=dict(seeds=list(SEEDS),train_query_origins=len(keys),extra_train_query_origins=len(extra),
                train_houses=len({k[0] for k in keys}),train_conflicts=len(tp),validation_houses=len({q[0] for q in ck}),
                criterion='Eligible nonzero caps must improve actual first hit in >=2/3 seeds, have nonnegative mean delta in >=3/6 houses, and clear +0.5pp average. Maximize gain minus 0.4 house SE; ties use swap then smaller step and cap. Zero cap falls back to original.',
                arms={}, matched={'step':320,'cap':.10},source_sha256=digest(Path(__file__)))
    old=previous(project)
    for arm in ARMS:
        grid=defaultdict(list)
        for seed in SEEDS:
            head=load_prior(old,project,seed,'jepa','cuda')
            prior,fore=prior_features(head,tr)
            pv,fv=prior_features(head,ca)
            model=train(seed,arm,tr,keys,tp,prior,fore,mean,folder)
            base=(pv+ca['memory']).cpu().numpy()
            before=balanced(ck,y,base.argmax(1))
            for step in STEPS:
                model.load_state_dict(torch.load(folder/f'{arm}_{seed}_{step}.pt',weights_only=True))
                r=predict(model.eval(),ca,fv).cpu().numpy()
                for cap in CAPS:
                    pick=np.argmax(base+cap*r,axis=1)
                    delta=balanced(ck,y,pick)-before
                    hit,contrast=measure(ck,y,pick,cp)
                    grid[step,cap].append((delta,hit,contrast))
        stats={};eligible=[]
        for key,rows in grid.items():
            delta=np.stack([r[0] for r in rows]); avg=delta.mean(0)
            hit,contrast=np.mean([r[1:] for r in rows],axis=0)
            util=avg.mean()-.4*avg.std(ddof=1)/np.sqrt(len(avg))
            ok=(key[1]==0 or (delta.mean(1)>0).sum()>=2 and (avg>=-1e-12).sum()>=3 and avg.mean()>=.005 and util>0)
            stats[f'{key[0]}/{key[1]}']=dict(hit=float(hit),swap=float(contrast),net=float(avg.mean()),
                                                regularized_gain=float(util),eligible=bool(ok))
            if ok:eligible.append((util,contrast,-key[0],-key[1],key))
        step,cap=max(eligible)[-1]
        report['arms'][arm]=dict(step=step,cap=cap,grid=stats,
            hashes={str(s):digest(folder/f'{arm}_{s}_{step}.pt') for s in SEEDS},
            matched_hashes={str(s):digest(folder/f'{arm}_{s}_320.pt') for s in SEEDS})
        print(arm,step,cap,stats[f'{step}/{cap}'],flush=True)
    write(lock,report)


@torch.inference_mode()
def evaluate(project,split,matched):
    out=project/'jepa_route_v12/refinement'
    lock=json.loads((out/'selection.json').read_text())
    keys,table=read(project,split)
    old=previous(project)
    dest=out/f'{split}{"_matched" if matched else ""}_choices.jsonl'
    pending=dest.with_suffix('.pending')
    with pending.open('w') as file:
        for seed in SEEDS:
            prior=load_prior(old,project,seed,'jepa','cuda')
            prob,fore=prior_features(prior,table)
            first=prob+table['memory']
            scores={'v11_jepa':first}
            for arm,meta in lock['arms'].items():
                config=lock['matched'] if matched else meta
                weights=out/'weights'/f'{arm}_{seed}_{config["step"]}.pt'
                assert digest(weights)==meta['matched_hashes' if matched else 'hashes'][str(seed)]
                state=torch.load(weights,weights_only=True)
                model=ForegroundResidual(table['text'].shape[-1],state['mean']).cuda()
                model.load_state_dict(state)
                scores[arm]=first+config['cap']*predict(model.eval(),table,fore)
            mem=table['memory'].cpu().tolist()
            for arm,data in scores.items():
                for i,row in enumerate(data.cpu().numpy()):
                    h,o,w=keys[i];file.write(json.dumps(dict(split=split,seed=seed,arm=arm,house=h,origin=o,
                        query=w,first=int(row.argmax()),score=row.tolist(),initial_memory=mem[i]))+'\n')
    pending.replace(dest)
    print('locked scores',dest.name,len(keys)*3*3,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('project',type=Path);p.add_argument('op',choices=('fit','calibration','fresh2'))
    p.add_argument('--matched',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if a.op=='fit':fit(a.project)
    else:evaluate(a.project,a.op,a.matched)
