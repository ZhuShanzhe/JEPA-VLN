"""Train on separate physical trees; lock update strength on calibration room forecast error."""
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scene_layout import EpisodeLayout


MODELS=('jepa','direct')
STRENGTHS=(0.,.25,.5,1.)


def bank(root,split,file):
    with np.load(file) as z:
        paths=z['paths'].tolist();plans=z['plans'];sensed=z['sensed'][:,:3].astype(float).mean(1)
        prior=z['prior'].astype(float);goals=z['goals']
        forecast={}
        for m in MODELS:
            p=z['env_pred_'+m].astype(float)
            if not np.allclose(p,p[:,:,:1],atol=1e-5):
                raise ValueError('room predictions unexpectedly depend on the target word')
            forecast[m]=p[:,:,0]
        if 'blocked' in z.files:blocked=z['blocked']
        else:
            base=root/'simulator/trees'/split/file.parent.name/file.stem
            # Load observed collision traces only. Policy features select each ancestor's own branch.
            bs=[]
            for path in paths:
                with np.load(base/('node_'+path+'.npz')) as trace:bs.append(trace['collided'])
            blocked=np.stack(bs)
    index={k:j for j,k in enumerate(paths)}
    outcome=np.zeros((len(paths),3,sensed.shape[-1]))
    valid=np.zeros(len(paths),bool)
    for i,path in enumerate(paths):
        if len(path)<4:
            valid[i]=True
            outcome[i]=[sensed[index[path+str(a)]] for a in range(3)]
    features={}
    for m,p in forecast.items():
        states={'r':EpisodeLayout()}
        x=np.zeros(p.shape+(7,),float)
        for path in sorted(paths,key=len):
            i=index[path]
            if path!='r':
                parent=path[:-1];j=index[parent];br=int(path[-1])
                state=copy.deepcopy(states[parent]);state.observe(p[:,j,br],plans[j,br],blocked[j,br],sensed[i])
                states[path]=state
            x[:,i]=states[path].features(p[:,i],plans[i])
        features[m]=x
    return paths,goals,prior,forecast,features,outcome,valid,sensed


def files(root,split):
    d=root/'horizon/banks' if split=='train' else root/'online/scored'/split
    group=sorted(d.glob('*/*.npz'))
    expected={'train':48,'calibration':48,'fresh':96}[split]
    if len(group)!=expected:raise RuntimeError(f'{split}: {len(group)} != {expected}')
    return group


def fit(root):
    blocks=defaultdict(list)
    for f in files(root,'train'):
        paths,_,_,p,features,y,mask,_=bank(root,'train',f)
        mask&=np.array([k!='r' for k in paths])
        for m in MODELS:
            x=features[m][:,mask].reshape(-1,7)
            target=(y[None,mask]-p[m][:,mask]).ravel()
            blocks[m,f.parent.name].append((x.T@x/len(x),x.T@target/len(x)))
    weights={}
    for m in MODELS:
        items=[v for (name,_),v in blocks.items() if name==m]
        gram=np.mean([np.mean([z[0] for z in v],0) for v in items],0)
        t=np.mean([np.mean([z[1] for z in v],0) for v in items],0)
        weights[m]=np.linalg.solve(gram+np.eye(len(t))*max(1e-6,.05*np.mean(np.diag(gram))),t)
    return weights


def normalize(p):
    p=p.clip(1e-6);return p/p.sum(-1,keepdims=True)


def evaluate(root,split,weights,gate=None):
    destination=root/'layout_update';book=defaultdict(list)
    for f in files(root,split):
        paths,goals,prior,p,features,y,mask,sensed=bank(root,split,f)
        mask&=np.array([k!='r' for k in paths])
        row={'paths':np.array(paths),'goals':goals}
        for m in MODELS:
            delta=features[m]@weights[m]
            for strength in STRENGTHS:
                fixed=p[m] if strength==0. else normalize(p[m]+strength*delta)
                if gate is not None and (strength==gate[m] or strength==1.):
                    diff=fixed-p[m]
                    affinity=np.einsum('snak,snqk->snqa',diff,prior).astype('float32')
                    if strength==gate[m]:row['adjust_'+m]=affinity
                    if strength==1.:row['fixed_'+m]=affinity
                before=fixed[:,mask];actual=y[None,mask]
                center=before-before.mean(2,keepdims=True)
                actual_center=actual-actual.mean(2,keepdims=True)
                persist=sensed[None,mask,None,:]
                metric=[((before-actual)**2).sum(-1).mean(),
                        ((center-actual_center)**2).sum(-1).mean(),
                        ((persist-actual)**2).sum(-1).mean()]
                book[m,strength,f.parent.name].append(metric)
        if gate is not None:
            out=destination/split/f.parent.name;out.mkdir(parents=True,exist_ok=True)
            tmp=out/(f.stem+'.pending.npz')
            np.savez_compressed(tmp,**row);tmp.replace(out/f.name)
    stats={}
    for m in MODELS:
        houses=sorted({h for n,_,h in book if n==m})
        by={g:np.array([np.mean(book[m,g,h],axis=0) for h in houses]) for g in STRENGTHS}
        stats[m]={}
        for g,val in by.items():
            d=val[:,0]-by[0.][:,0];bootstrap=np.random.default_rng(1129).choice(d,(12000,len(d))).mean(1)
            stats[m][str(g)]=dict(room_brier=float(val[:,0].mean()),route_contrast_brier=float(val[:,1].mean()),
                copy_current_room=float(val[:,2].mean()),delta_vs_base=float(d.mean()),
                ci95=np.quantile(bootstrap,[.025,.975]).tolist(),houses=len(houses),
                improved_houses=int((d<0).sum()))
    return stats


def main(root):
    dest=root/'layout_update'
    weights=fit(root)
    cal=evaluate(root,'calibration',weights)
    # Household-equal score, identical to the training target. Ties prefer no change.
    gate={m:min(STRENGTHS,key=lambda g:cal[m][str(g)]['room_brier']) for m in MODELS}
    record=dict(weights={m:w.tolist() for m,w in weights.items()},strength=gate,calibration=cal,
            frozen_criterion='minimum calibration soft-label squared room error; smallest change on tie',
            train_houses=12,train_roots=48,train_edges_after_arrival=48*36)
    lock=dest/'frozen.json'
    if lock.exists() and json.loads(lock.read_text())!=record:raise ValueError('selection already frozen')
    lock.write_text(json.dumps(record,indent=2))
    for s in ('calibration','fresh'):
        summary=evaluate(root,s,weights,gate)
        (dest/(s+'_summary.json')).write_text(json.dumps(summary,indent=2))
    print(gate,flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
