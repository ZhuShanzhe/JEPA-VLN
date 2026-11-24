"""Bound short-step forecasts with the expected value of continuing an actual search."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve


def state_features(score,env,prior,sensed,cue,plans,remaining,past_room,past_cue):
    """Only current observations, committed route forecasts and the visited prefix."""
    sensed=np.asarray(sensed);context=sensed[:3].mean(0)
    turn=np.cumsum((plans==1).astype(float)-(plans==2),axis=-1)*np.pi/6
    walk=(plans==0)*.25
    dx=(np.cos(turn)*walk).sum(-1);dy=(np.sin(turn)*walk).sum(-1)
    affinity=env@prior
    overlap=env@context;previous=env@past_room
    dirs=np.argmax(np.cos(np.arctan2(dy,dx)[:,None]-np.deg2rad([-30.,0.,30.])),axis=-1)
    anglecue=cue[dirs]
    left=max(0,remaining-1)/3.
    scalars=np.stack([score,affinity,overlap,previous,env.max(1),-(env*np.log(env.clip(1e-9))).sum(-1),
                   np.hypot(dx,dy)/6,dx/6,dy/6,anglecue,
                   affinity*(1-overlap),affinity*(1-previous),score*affinity,
                   (previous-overlap)*past_cue,anglecue*affinity],axis=-1)
    return np.concatenate([scalars,scalars*left,env,env*prior,env*(1-past_room)],axis=-1)


def features(bank,arm,seed,idx,q,horizon=None):
    paths=bank['paths'].tolist();lookup={p:i for i,p in enumerate(paths)}
    path=paths[idx];prior_ids=[lookup[path[:k]] for k in range(1,len(path))]
    cur=bank['sensed'][idx,:3]
    history=(np.mean(bank['sensed'][prior_ids,:3],axis=(0,1)) if prior_ids else cur.mean(0))
    clue=(float(np.mean(bank['cue'][prior_ids,:3,q])) if prior_ids else float(np.mean(bank['cue'][idx,:3,q])))
    return state_features(bank[arm][seed,idx,q],bank['env_pred_'+arm][seed,idx,q],
            bank['prior'][seed,idx,q],cur,bank['cue'][idx,:3,q],bank['plans'][idx],
            5-len(path) if horizon is None else horizon,history,clue)


def trace_sets(truth,word):
    target={int(i) for i,w in truth['instance_words'].items() if w==word}
    visits={}
    for p,n in truth['nodes'].items():
        scan=set().union(*n['scan'].values())&target
        visits[p]=[scan|(set().union(*w.values())&target) for w in n['walk']]
    return visits


def label(paths,visits,goal_quota,current_scan):
    out=[]
    quota=max(1,min(2,goal_quota))
    for i,path in enumerate(paths):
        remaining=5-len(path)
        if remaining<2:continue
        ancestors=set()
        for k in range(1,len(path)):ancestors.update(visits[path[:k]][int(path[k])])
        # Instance appearances at the current common entrance are already observed
        # when choosing. Subtract only simulator-confirmed shared scan IDs.
        ancestors.update(current_scan[path])
        for budget in range(2,remaining+1):
            def options(p,depth,seen):
                gains=[]
                for k,view in enumerate(visits[p]):
                    known=seen|view
                    if depth==1:
                        gains.append(min(len(known-ancestors),max(0,quota-len(ancestors)))/quota)
                    else:
                        child=options(p+str(k),depth-1,known)
                        gains.append(.8*child.mean()+.2*child.max())
                return np.array(gains)
            y=options(path,budget,ancestors)
            if np.ptp(y)>1e-8:
                out.append((i,budget,y))
    return out


def bank_dict(path):
    with np.load(path) as f:return {k:f[k] for k in f.files}


def fit(root,arm):
    xs=[];ys=[];houses=[]
    for f in sorted((root/'horizon/banks').glob('*/*.npz')):
        h=f.parent.name;home=root/'simulator/trees/train'/h/f.stem
        b=bank_dict(f);paths=b['paths'].tolist()
        truth=json.loads((home/'truth.json').read_text())
        goals=json.loads((home/'training_targets.json').read_text())['targets']
        for q,g in enumerate(goals):
            visits=trace_sets(truth,g['word'])
            ids={int(j) for j,word in truth['instance_words'].items() if word==g['word']}
            scans={p:set().union(*n['scan'].values())&ids for p,n in truth['nodes'].items()}
            for i,budget,y in label(paths,visits,g['quota'],scans):
                for seed in range(3):
                    x=features(b,arm,seed,i,q,budget)
                    xs.extend(x-x.mean(0));ys.extend(y-y.mean());houses.extend([h]*3)
    x=np.stack(xs);y=np.array(ys);hh=np.asarray(houses)
    count={h:(hh==h).sum() for h in set(hh)}
    w=np.asarray([1/(len(count)*count[h]) for h in hh])
    scale=np.maximum(np.sqrt(w@(x*x)),.025)
    a=x/scale;gram=a.T@(w[:,None]*a)
    coef=solve(gram+.125*np.eye(gram.shape[0]),a.T@(w*y),assume_a='pos')
    return dict(scale=scale.tolist(),coef=coef.tolist(),training_houses=len(count),informative_choices_per_seed=len(x)//9)


class ValueResidual:
    def __init__(self,settings):
        self.settings=settings

    @classmethod
    def load(cls,project,arm='jepa'):
        path=Path(project)/'jepa_env_v13/horizon/selection.json'
        return cls(json.loads(path.read_text())[arm])

    def route_delta(self,score,env,prior,sensed,cue,plans,remaining,past_room,past_cue):
        """Call after scoring the current junction, before inspecting a candidate."""
        x=state_features(score,env,prior,sensed,cue,plans,remaining,past_room,past_cue)
        return self.adjust(x,remaining)

    def adjust(self,x,remaining):
        if remaining<=1:return np.zeros(len(x))
        a=(x-x.mean(0))/self.settings['scale']
        return self.settings['strength']*np.clip(a@self.settings['coef'],-.35,.35)


def deltas(bank,arm,settings):
    head=ValueResidual(settings)
    paths=bank['paths'].tolist();d=np.zeros_like(bank[arm])
    for seed in range(3):
        for i,p in enumerate(paths):
            if len(p)>3:continue
            for q in range(len(bank['goals'])):d[seed,i,q]=head.adjust(features(bank,arm,seed,i,q),5-len(p))
    return d


def calibration(root,arm,head):
    family=defaultdict(lambda:defaultdict(list))
    for f in sorted((root/'online/scored/calibration').glob('*/*.npz')):
        h=f.parent.name;home=root/'simulator/trees/calibration'/h/f.stem
        bank=bank_dict(f);paths={v:i for i,v in enumerate(bank['paths'].tolist())}
        truth=json.loads((home/'truth.json').read_text());goals=json.loads((home/'episode.json').read_text())['targets']
        residual=deltas(bank,arm,dict(head,strength=1.))
        for q,g in enumerate(goals):
            if g['quota']==0:continue
            visits=trace_sets(truth,g['word'])
            for strength in (0.,.25,.5,1.,1.5):
                rates=[]
                for seed in range(3):
                    p='r';seen=set()
                    for depth in range(4):
                        i=paths[p];s=int(np.argmax(bank[arm][seed,i,q]+strength*residual[seed,i,q]))
                        seen.update(visits[p][s]);p+=str(s)
                    rates.append(min(1.,len(seen)/g['quota']))
                family[strength][h].append(np.mean(rates))
    table={strength:float(np.mean([np.mean(samples) for samples in houses.values()])) for strength,houses in family.items()}
    return min(table,key=lambda k:(-table[k],k)),table


def main(root):
    assert len(list((root/'horizon/banks').glob('*/*.npz')))==48
    d=root/'horizon';dst=d/'selection.json'
    if dst.exists():out=json.loads(dst.read_text())
    else:
        out={}
        for arm in ('jepa','direct'):
            head=fit(root,arm)
            strength,table=calibration(root,arm,head)
            out[arm]=dict(head,strength=strength,calibration=table)
            print(arm,strength,table,flush=True)
        dst.write_text(json.dumps(out,indent=2))
    for split in ('calibration','fresh'):
        for f in sorted((root/'online/scored'/split).glob('*/*.npz')):
            b=bank_dict(f);dst=d/'residual'/split/f.parent.name/f.name;dst.parent.mkdir(parents=True,exist_ok=True)
            tmp=dst.with_suffix('.partial.npz')
            np.savez_compressed(tmp,paths=b['paths'],goals=b['goals'],residual=deltas(b,'jepa',out['jepa']),
                    residual_direct=deltas(b,'direct',out['direct']))
            tmp.replace(dst)
        print(split,'done',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('experiment',type=Path);main(p.parse_args().experiment)
