"""House-balanced pairwise fitting and locked evaluation of the causal route scorer."""
from collections import defaultdict
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from v10_live import FEATURES,DEFAULT,prior_transform

D=Path(__file__).resolve().parent
SRC=D.parents[1]/'jepa_separation_v8_20261123/primitives'
H=(4,12,24)
BOUNDS=((.05,.30),(0,.15),(.035,.40),(.035,.50),(.02,.80))
ARMS=('full','fixed','observed','geometry','no_semantic_memory','wrong_goal','v8')
def read(split):
    rows=json.loads((SRC/(split+'.index.json')).read_text())
    for r in rows:
        p=D.parent/'features'/(hashlib.sha256(r['sid'].encode()).hexdigest()+'.npz')
        with np.load(p) as z:
            yield r['house'],r['sid'],{k:z[k] for k in z.files}
def design(f):
    return np.concatenate([prior_transform(f[...,[0,2]]),f[...,3:6]],-1)
def targetpairs(a,y):
    distinct=(a[:,None]!=a[None,:]).any(-1);idx,jdx=np.triu(distinct,k=1).nonzero()
    good=(y[idx]!=y[jdx]);return idx,jdx,good

def train():
    byhouse=defaultdict(list)
    for house,sid,z in read('train'):
        for hi,h in enumerate(H):
            f=design(z['f'][hi]);y=z['y'][hi]
            i,j,mask=targetpairs(z['actions'][:,:h],y)
            sign=np.sign(y[i].astype(float)-y[j].astype(float))
            delta=(f[i]-f[j])*sign[:,:,None]
            keep=mask & z['present'][None].astype(bool)
            if keep.any():byhouse[house].append(delta[keep])
    groups=[np.concatenate(g) for g in byhouse.values()];p0=np.array([DEFAULT[k] for k in FEATURES])
    # Balanced informative house populations, fixed range / temperature / prior.
    def loss(w):
        val=.24*((w-p0)**2).sum();grad=.48*(w-p0)
        for group in groups:
            l=6.*(group@w);val+=np.logaddexp(0,-l).mean()/len(groups)
            grad+=(-6.*group/(1+np.exp(l[:,None]))).mean(0)/len(groups)
        return val,grad
    fit=minimize(loss,p0,method='L-BFGS-B',jac=True,bounds=BOUNDS)
    if not fit.success:raise RuntimeError(fit.message)
    ck=dict(schema=1,weights=dict(zip(FEATURES,fit.x.tolist())),fit="house-balanced pairwise constrained logistic readout; frozen V8 action-conditioned JEPA predictor",
            initial=dict(zip(FEATURES,p0)),bounds=dict(zip(FEATURES,BOUNDS)),temperature=6.,regularization=.24,
            train_houses=list(sorted(byhouse)),informative_pairs=int(sum(map(len,groups))),initial_loss=float(loss(p0)[0]),final_loss=float(fit.fun),
            names=list(FEATURES),seed=0)
    path=D/'v10.json';payload=json.dumps(ck,indent=2,allow_nan=False)
    if path.exists() and path.read_text()!=payload:raise FileExistsError('different frozen V10 checkpoint')
    if not path.exists():
        tmp=path.with_suffix('.pending');tmp.write_text(payload);tmp.replace(path)
    print(json.dumps(ck),flush=True)

def eval_split(split,w):
    allh=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    opportunities=defaultdict(int);queryby=defaultdict(lambda:defaultdict(list));changed=defaultdict(lambda:defaultdict(list))
    meta=defaultdict(int)
    for house,sid,z in read(split):
        meta['origins']+=1;meta['origins_with_real_translation']+=int(z['translated'])
        meta['executed_prefix_actions']+=int(z['steps']-1);meta['real_prefix_rgb_keyframes']+=int(z['keyframes'])
        for hi,h in enumerate(H):
            f=z['f'][hi];y=z['y'][hi];d=design(f);a=z['actions'][:,:h]
            p,g,novel,hazard,trace=np.moveaxis(d,-1,0);o=prior_transform(f[:,:,1])
            s=dict(fixed=w[0]*p+w[1]*g,v8=f[:,:,0])
            s['geometry']=w[1]*g+w[2]*novel+w[3]*hazard
            s['no_semantic_memory']=s['fixed']+w[2]*novel+w[3]*hazard
            s['full']=s['no_semantic_memory']+w[4]*trace
            s['observed']=w[0]*o+s['geometry']+w[4]*trace
            # Deterministic vocabulary rotation independent of scene/query availability.
            s['wrong_goal']=np.roll(s['full'],shift=-1,axis=1)
            i,j,inform=targetpairs(a,y)
            present=np.asarray(z['present'],bool)
            for q in np.flatnonzero(present):
                target=y[:,q];maximum=float(target.max());minimum=float(target.min())
                meta['present_goal_origin_horizons']+=1
                if maximum>0:opportunities['any_feasible_gain']+=1
                if maximum<=minimum:continue
                opportunities['different_candidate_outcomes']+=1
                base=int(s['fixed'][:,q].argmax());full=int(s['full'][:,q].argmax())
                nosem=int(s['no_semantic_memory'][:,q].argmax())
                changed[house]['action_differs_from_fixed'].append(bool(np.any(a[base]!=a[full])))
                changed[house]['semantic_memory_changes_action'].append(bool(np.any(a[nosem]!=a[full])))
                changed[house]['choice_improves_vs_fixed'].append(bool(target[full]>target[base]))
                changed[house]['choice_regresses_vs_fixed'].append(bool(target[full]<target[base]))
                for arm,v in s.items():
                    k=int(v[:,q].argmax());gain=float(target[k]);delta=(v[i,q]-v[j,q])*np.sign(target[i].astype(float)-target[j].astype(float))
                    delta=delta[inform[:,q]]
                    rec=allh[house][arm];rec['top_gain'].append(gain);rec['any_gain'].append(float(gain>0))
                    rec['selects_best'].append(float(gain==maximum));rec['regret'].append(maximum-gain)
                    rec['pair_accuracy'].append(float(np.mean((delta>1.e-8)+.5*(np.abs(delta)<=1.e-8))))
                    queryby[str(z['words'][q])][arm].append(gain)
    metrics={hs:{a:{k:float(np.mean(xx)) for k,xx in v.items()} for a,v in arms.items()} for hs,arms in allh.items()}
    means={a:{k:float(np.mean([d[a][k] for d in metrics.values()])) for k in metrics[next(iter(metrics))][a]} for a in ARMS}
    changes={k:float(np.mean([np.mean(v[k]) for v in changed.values()])) for k in next(iter(changed.values()))}
    ci={}
    for arm in ['fixed','observed','no_semantic_memory','geometry','wrong_goal','v8']:
        diff=np.array([xx['full']['top_gain']-xx[arm]['top_gain'] for xx in metrics.values()])
        boot=np.random.default_rng(971).choice(diff,(10000,len(diff)),replace=True).mean(1)
        ci[arm]={'mean':float(diff.mean()),'house_resampling_95pct':np.quantile(boot,[.025,.975]).tolist()}
    return dict(coverage=meta,opportunities=opportunities,evaluated_houses=list(sorted(metrics)),
                house_macro=means,selection_changes_house_macro=changes,full_minus_reference_mean_top_gain=ci,
                per_house=metrics,per_goal_pooled_top_gain={q:{a:float(np.mean(s)) for a,s in v.items()} for q,v in queryby.items()})

def evaluate():
    ck=json.loads((D/'v10.json').read_text());w=np.array([ck['weights'][k] for k in FEATURES])
    frozen=hashlib.sha256((D/'v10.json').read_bytes()).hexdigest()
    splits={s:eval_split(s,w) for s in ['train','calibration','holdout']}
    cohorts=[set(d['evaluated_houses']) for d in splits.values()]
    assert all(not a.intersection(b) for i,a in enumerate(cohorts) for b in cohorts[i+1:])
    obj=dict(checkpoint_sha256=frozen,outcome='newly eligible instance-viewpoint opportunities along a physical candidate. Scores are on present goal/origin/horizons with differing realized candidate gains; this is not end-to-end success.',
             inputs='single final RGB for frozen V8 prior; entire executed supplied prefix RGB/odometry for memory; no semantic IDs; actual return validated only for collection',
             arms=list(ARMS),splits=splits)
    path=D/'hssd_results.json';path.write_text(json.dumps(obj,indent=2,allow_nan=False));print('DONE',frozen,splits['holdout']['house_macro'],flush=True)
if __name__=='__main__':
    if sys.argv[1]=='train':train()
    elif sys.argv[1]=='test':evaluate()
    else:raise ValueError(sys.argv[1])
