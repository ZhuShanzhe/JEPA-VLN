"""Train and calibrate action-conditioned probability of new catalogue viewpoint eligibility."""
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D=Path(__file__).resolve().parent
ROOT=D.parents[2]
V10=D.parents[1]
H=(4,12,24)
OFFICIAL={'chair':'chair.n.01','bed':'bed.n.01','couch':'sofa.n.01','toilet':'toilet.n.02',
          'potted_plant':'potted_plant.n.01','tv':'television_receiver.n.01'}
def rlogit(p):return logit(np.clip(p,1e-5,1-1e-5))
def fields(score,geometry,visual_trace,acts):
    """One route, all goals. Exploration rewards and opportunity labels are not inputs."""
    h=len(acts);yaw=np.cumsum((acts==1).astype(float)-(acts==2))*np.pi/6;f=acts==0
    displacement=.25*np.hypot(np.sum(np.cos(yaw)*f),np.sum(np.sin(yaw)*f))
    z=rlogit(score);budget=np.full_like(z,h/24.)
    return np.stack([z,rlogit(geometry),visual_trace,np.full_like(z,displacement),np.full_like(z,f.mean()),
                    budget,z*(h==4),z*(h==24),visual_trace*budget,z*visual_trace],-1)
def collect(split):
    src=ROOT/'jepa_separation_v8_20261123/primitives'
    rec=json.loads((src/(split+'.index.json')).read_text())
    cols={k:[] for k in ['joint','observed','house','sid','q','y','h','catalog_present']}
    counts={'origins':len(rec),'nominal_candidate_query_rows':0,'removed_duplicate_prefix_rows':0,'excluded_unresolved_catalog_rows':0,
            'num_sequences':0,'duplicate_physical_label_disagreements':0,'duplicate_max_feature_difference':0.}
    for r in rec:
        fn=V10/'features'/(hashlib.sha256(r['sid'].encode()).hexdigest()+'.npz')
        with np.load(fn) as f:words=f['words'];present=np.asarray(f['present'],bool);labels=f['y'];cache=f['f'];acts=f['actions']
        absent=np.isin(words,[OFFICIAL[k] for k in r['catalog_exclusions']]);ok=~absent;K=int(ok.sum())
        for hi,h in enumerate(H):
            counts['nominal_candidate_query_rows']+=len(acts)*len(words)
            seen={}
            for a,act in enumerate(acts[:,:h]):
                key=tuple(act.tolist())
                if key in seen:
                    i=seen[key]
                    counts['removed_duplicate_prefix_rows']+=len(words)
                    counts['duplicate_physical_label_disagreements']+=int(np.sum(labels[hi,i]!=labels[hi,a]))
                    md=float(abs(cache[hi,i]-cache[hi,a]).max())
                    counts['duplicate_max_feature_difference']=max(counts['duplicate_max_feature_difference'],md)
                    continue
                seen[key]=a;counts['num_sequences']+=1
                counts['excluded_unresolved_catalog_rows']+=int(absent.sum())
                ff=cache[hi,a]
                for mode,idx in [('joint',0),('observed',1)]:
                    cols[mode].append(fields(ff[:,idx],ff[:,2],ff[:,5],act)[ok])
                for key,vec in [('house',np.repeat(r['house'],K)),('sid',np.repeat(r['sid'],K)),('q',words[ok]),
                                ('y',labels[hi,a,ok]>0),('h',np.repeat(h,K)),('catalog_present',present[ok])]:cols[key].append(vec)
    if counts['duplicate_physical_label_disagreements'] or counts['duplicate_max_feature_difference']>5e-6:
        raise ValueError(('inconsistent duplicate nominal prefix',counts))
    return {k:np.concatenate(v) for k,v in cols.items()},counts
def weights_by_house(house):
    names,ids,c=np.unique(house,return_inverse=True,return_counts=True);return len(house)/len(names)/c[ids]
def fit_raw(z,mode):
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.4,max_iter=800,random_state=197))
    w=weights_by_house(z['house'])
    model.fit(z[mode],z['y'],standardscaler__sample_weight=w,logisticregression__sample_weight=w)
    return model
def calibrate(raw,z):
    # One monotone Platt slope, three horizon intercepts; shrink to original log odds.
    w=weights_by_house(z['house']);y=z['y'].astype(float);hot=np.array([z['h']==h for h in H]).T
    a=np.column_stack((raw,hot));prior=np.array([1.,0.,0.,0.]);reg=np.array([.003,.001,.001,.001])
    def fun(beta):
        logits=a@beta
        L=np.average(np.logaddexp(0.,logits)-y*logits,weights=w)+np.sum(reg*(beta-prior)**2)
        g=a.T@(w*(expit(logits)-y))/w.sum()+2*reg*(beta-prior)
        return L,g
    sol=minimize(fun,prior,jac=True,method='L-BFGS-B',bounds=[(.03,8.)]+[(-8.,8.)]*3)
    if not sol.success:raise RuntimeError(sol.message)
    return sol.x
def predict(model,beta,ff,h):
    hh=np.asarray(h)
    offsets=sum(b*(hh==n) for n,b in zip(H,beta[1:]))
    return expit(beta[0]*model.decision_function(ff)+offsets)
def stats(y,p):
    y=y.astype(float);n=len(y)
    return dict(n=n,positives=int(y.sum()),negative=n-int(y.sum()),prevalence=float(y.mean()),
        average_precision=float(average_precision_score(y,p)) if y.any() else None,
        brier=float(np.mean((y-p)**2)),mean_prediction=float(p.mean()))
def evalset(z,models,freq):
    pred={'horizon_frequency':np.array([freq[h] for h in z['h']])}
    for mode,(m,b) in models.items():pred[mode]=predict(m,b,z[mode],z['h'])
    out={}
    for part,mask in [('all',np.ones(len(z['y']),bool))]+[(str(h)+'_actions',z['h']==h) for h in H]+[('catalog_contains_target_class',z['catalog_present'])]:
        out[part]={k:stats(z['y'][mask],p[mask]) for k,p in pred.items()}
    out['per_house_all']={s:{k:stats(z['y'][z['house']==s],p[z['house']==s]) for k,p in pred.items()} for s in np.unique(z['house'])}
    return out
def main():
    train,train_a=collect('train');cal,cal_a=collect('calibration')
    if set(train['house'])&set(cal['house']):raise ValueError('training/calibration overlap')
    models={}
    for name in ('joint','observed'):
        m=fit_raw(train,name);models[name]=m,calibrate(m.decision_function(cal[name]),cal)
    ww=weights_by_house(cal['house'])
    freq={h:float((np.sum(ww[cal['h']==h]*cal['y'][cal['h']==h])+.5)/(np.sum(ww[cal['h']==h])+1.)) for h in H}
    ck=D/'opportunity_calibrators.joblib';joblib.dump(dict(models=models,horizon_frequency=freq),ck)
    hold,hol_a=collect('holdout')
    if set(hold['house'])&(set(train['house'])|set(cal['house'])):raise ValueError('holdout overlap')
    out=dict(target='P(at least one newly eligible catalogued goal-instance viewpoint within the specified physical action budget | legal action prefix, RGB evidence, recent visual trace, frozen V8 features)',
        task='HSSD-derived branch opportunity; all 16 query classes scored, including truly absent classes; corrupt category catalogues excluded. No class-presence flag or exploration bonus is a predictor input.',
        sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),counts={'train':train_a,'calibration':cal_a,'holdout':hol_a},
        splits={s:evalset(z,models,freq) for s,z in [('train',train),('calibration',cal),('holdout',hold)]})
    (D/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False));print('DONE',out['sha256'],out['splits']['holdout']['all'])
if __name__=='__main__':main()
