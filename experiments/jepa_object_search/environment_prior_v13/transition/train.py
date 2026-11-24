"""Train on forty houses; freeze the change correction using only the six calibration houses."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from transition import ChangeNet,features


def collect(project,base,splits):
    out=defaultdict(list);d=project/'jepa_env_v13/data'
    for split in splits:
        source=project/('jepa_route_v12' if split=='extra_train' else 'jepa_hidden_v11')
        for p in sorted((source/'models/cache/hssd'/split/'actor').glob('*.npz')):
            with np.load(p) as actor:
                h=str(actor['house']);o=str(actor['origin'])
                x,dep,orig,idx=features(base,actor['ctx'],actor['actions'])
            with np.load(d/'labels'/split/(o+'.npz')) as lab:
                target=lab['view_room']; start=lab['start_view_room'][idx]
            ok=(target.sum(-1)>.9)&(start.sum(-1)>.9)
            if not ok.any():continue
            for key,val in [('x',x.cpu().numpy()),('departure',dep.cpu().numpy()),('original',orig.cpu().numpy()),
                            ('y',target),('change',target.argmax(-1)!=start.argmax(-1)),
                            ('house',np.full(idx.shape,h)),('tap',np.broadcast_to(np.arange(4),idx.shape))]:
                out[key].append(val[ok])
    return {k:np.concatenate(v) for k,v in out.items()}


def metrics(data,prob):
    y=data['y'];h=data['house'];change=data['change']
    brier=np.sum((prob-y)**2,axis=-1); correct=prob.argmax(-1)==y.argmax(-1)
    result={}
    for k,mask in [('all',np.ones(len(y),bool)),('change',change),('same',~change)]:
        by=[(brier[mask&(h==i)].mean(),correct[mask&(h==i)].mean()) for i in np.unique(h[mask])]
        result[k]=np.mean(by,axis=0).tolist()
    return result


def train(project,residual):
    d=project/'jepa_env_v13';sys.path.insert(0,str(d/'models'))
    from env_routes import EnvironmentRoutes
    stats=defaultdict(list); widths=[]
    for seed in [11,29,47]:
        base=EnvironmentRoutes(project,seed)
        tr=collect(project,base,['train','extra_train']);val=collect(project,base,['calibration'])
        truth=tr['y'].argmax(-1);condition=tr['change'].astype(int)
        weights=np.ones(len(truth))
        # Equalize the four horizons and the two transition outcomes within each,
        # then gently increase the frequency of rare destinations and cap extremes.
        for t in range(4):
            for c in range(2):
                mask=(tr['tap']==t)&(condition==c);weights[mask]/=max(1,mask.sum())
        freq=np.bincount(truth,minlength=tr['y'].shape[1])
        weights*=np.sqrt(freq.max()/np.maximum(1,freq[truth])).clip(1,3)
        h=tr['house']
        for house in np.unique(h):
            mask=h==house;weights[mask]/=weights[mask].sum()
        weights/=weights.sum()
        rng=np.random.default_rng(seed);torch.manual_seed(seed)
        xdim=tr['x'].shape[1];widths.append(xdim)
        net=ChangeNet(xdim,tr['y'].shape[1],residual=residual).to(base.device)
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.03)
        tensors={k:base.tensor(tr[k]) for k in ['x','departure','original','y','change']}
        td=d/'transition'/'residual' if residual else d/'transition'
        td.mkdir(exist_ok=True)
        for t in range(1,401):
            ix=rng.choice(len(truth),128,replace=True,p=weights)
            x,departure,y,ch=(tensors[k][ix] for k in ['x','departure','y','change'])
            pred,event,rooms=net(x,departure,tensors['original'][ix])
            loss=-(y*pred.clamp_min(1e-8).log()).sum(-1).mean()+.4*F.binary_cross_entropy_with_logits(event,ch)
            if residual:
                loss+=((pred-y)**2).sum(-1).mean()
            elif ch.any():loss-=.5*(y[ch.bool()]*rooms[ch.bool()].log_softmax(-1)).sum(-1).mean()
            opt.zero_grad();loss.backward();opt.step()
            if t in (100,250,400):
                ck=td/f'seed_{seed}_step_{t}.pt';torch.save(net.state_dict(),ck)
                with torch.inference_mode():
                    net.eval();p=net(base.tensor(val['x']),base.tensor(val['departure']),base.tensor(val['original']))[0].cpu().numpy();net.train()
                for mix in (0.,.2,.5,.8):
                    stats[t,mix].append(metrics(val,(1-mix)*val['original']+mix*p))
    averaged={k:{s:np.mean([v[s] for v in rows],axis=0).tolist() for s in rows[0]} for k,rows in stats.items()}
    baseline=averaged[100,0.]
    best=(100,0.);obj=lambda v:.5*(v['all'][0]+v['change'][0])
    for k,v in averaged.items():
        if v['all'][0]<=baseline['all'][0]+.002 and v['change'][0]<baseline['change'][0] and obj(v)<obj(averaged[best]):
            best=k
    step,mix=best
    summary=dict(selected=dict(step=step,mix=mix),input_dim=widths[0],train_frames=len(truth),calibration_frames=len(val['y']),
        train_changes=int(condition.sum()),calibration_changes=int(val['change'].sum()),
        metric_order=['brier','accuracy'],baseline=baseline,selected_metrics=averaged[best],
        calibration_grid={str(k):v for k,v in averaged.items()},
        sha256={str(s):hashlib.sha256((td/f'seed_{s}_step_{step}.pt').read_bytes()).hexdigest() for s in [11,29,47]})
    (td/'selection.json').write_text(json.dumps(summary,indent=2))
    print(summary['selected'],baseline,averaged[best],flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('project',type=Path);a.add_argument('--residual',action='store_true')
    opts=a.parse_args();train(opts.project,opts.residual)
