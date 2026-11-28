"""Fit target-size regression and clear-frame probabilities; choose fusion on disjoint calibration houses."""
from collections import Counter
from hashlib import sha256
import json
import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from clarity_refit import HERE,PRIOR,blend
from prepare_clarity import all_files
from frame_confirm import Trees,thresholds
from fit_confirm import pack

THRESH=thresholds()['threshold']


def get(split):
    xx=[];yy=[];groups=[];houses=[];counts=Counter()
    for k,(_,dest) in enumerate(all_files()):
        if dest.parent.parent.name!=split:continue
        with np.load(dest) as d:
            x=d['x'];info=d['labels'];area=int(d['source_box_area'])
            counts['camera_samples']+=int(d['frames'])
            counts['motion_sweeps']+=1
            counts['source_occupies_under_20pct']+=int(int(d['source_pixels'])<.2*area)
            if not len(x):continue
        xx.extend(x);yy.extend(info);groups.extend([k]*len(x));houses.extend([dest.parent.name]*len(x))
    x,y,g,h=map(np.asarray,(xx,yy,groups,houses))
    good=(y[:,1]>=369)&y[:,2].astype(bool)
    counts['usable_current_crops']=len(y)
    counts['clear_frames_missed_by_legacy']=int(np.sum(good&(y[:,3]*y[:,4]<THRESH)))
    counts['below_369_pixels']=int(np.sum(y[:,1]<369))
    counts['under_369_and_area_overestimated']=int(np.sum((y[:,1]<369)&(y[:,5]>=369)))
    counts['wrong_instance_or_bad_box']=int(np.sum(~y[:,2].astype(bool)))
    return x,y,g,h,dict(counts)


def classify(p,info,groups):
    score=p*info[:,4];good=(info[:,1]>=369)&(info[:,2]>0)
    chosen=np.flatnonzero(score>=THRESH)
    _,first=np.unique(groups[chosen],return_index=True);first=chosen[first]
    return dict(accepted_frames=len(chosen),bad_frames=int((~good[chosen]).sum()),
                first_correct=int(good[first].sum()),first_bad=int((~good[first]).sum()))


def export_area(model,path,x):
    trees=[e[0].tree_ for e in model.estimators_]
    n=max(t.node_count for t in trees)
    data={k:np.zeros((len(trees),n),dtype=float if k in ('threshold','value') else int) for k in ('threshold','value','key','left','right')}
    for i,t in enumerate(trees):
        for k,v in [('key',t.feature),('left',t.children_left),('right',t.children_right),('threshold',t.threshold),('value',t.value[:,0,0])]:
            data[k][i,:t.node_count]=v
    np.savez_compressed(path,depth=model.max_depth,rate=model.learning_rate,initial=float(model.init_.constant_.ravel()[0]),**data)
    assert np.allclose(Trees(path)(x[:100]),1/(1+np.exp(-model.predict(x[:100]))))


def regressor():
    return GradientBoostingRegressor(n_estimators=125,max_depth=3,min_samples_leaf=15,learning_rate=.055,random_state=381)


def main():
    os.umask(0o077)
    x,y,g,h,tra=get('train');vx,vy,vg,vh,val=get('val')
    assert len(set(h))==12 and len(set(vh))==6 and not set(h)&set(vh)
    directory=HERE/'private_clarity_model';directory.mkdir(exist_ok=True)
    area=y[:,1];target=np.log((np.clip(area,0,6000)+1)/370)
    # Upweight the clearance boundary where pixel estimation affects the action.
    weights=1+.6*((area>=160)&(area<=850))
    oof=np.zeros(len(y))
    for holdout in np.array_split(np.unique(h),3):
        hold=np.isin(h,holdout)
        m=regressor().fit(x[~hold],target[~hold],sample_weight=weights[~hold]);oof[hold]=m.predict(x[hold])
    logistic=LogisticRegression(C=5,random_state=381).fit(oof[:,None],area>=369)
    slope=float(logistic.coef_[0,0]);intercept=float(logistic.intercept_[0])
    area_model=regressor().fit(x,target,sample_weight=weights);export_area(area_model,directory/'pixel_area.npz',x)
    logits=area_model.predict(vx);ap=logistic.predict_proba(logits[:,None])[:,1]
    classifier=GradientBoostingClassifier(n_estimators=118,max_depth=3,min_samples_leaf=15,learning_rate=.065,random_state=381).fit(x,area>=369,sample_weight=weights)
    pack(classifier,directory/'clear_frame.npz',x);cp=classifier.predict_proba(vx)[:,1]
    old=classify(vy[:,3],vy,vg);choices=[];summaries=[]
    # Maintain prior first-accept errors and frame precision on calibration; keep fixed final threshold.
    for w in (0.,.5,1.):
        for floor in (0.,.85,1.):
            p=blend(vy[:,3],cp,ap,w,floor);stats=classify(p,vy,vg)
            fp=stats['bad_frames']/max(stats['accepted_frames'],1)
            choices.append(((stats['first_bad']<=old['first_bad'] and fp<=old['bad_frames']/max(1,old['accepted_frames']),
                             -max(0,stats['first_bad']-old['first_bad']),stats['first_correct'],-stats['first_bad'],-stats['bad_frames'],w,-floor),w,floor,stats))
            summaries.append(dict(area_weight=w,old_floor=floor,**stats))
    key,w,floor,stats=max(choices)
    cfg=dict(area_weight=w,old_floor=floor,slope=slope,intercept=intercept)
    (directory/'calibration.json').write_text(json.dumps(cfg)+'\n')
    pix=np.maximum(0,370*np.exp(logits)-1)
    usable=(vy[:,2]==1)&(vy[:,1]>0)
    miss=usable&(vy[:,1]>=369)&(vy[:,3]*vy[:,4]<THRESH)
    newp=blend(vy[:,3],cp,ap,w,floor)
    checks={subset:{'frames':int(mask.sum()),'old_mae_pixels':float(np.mean(np.abs(vy[mask,5]-vy[mask,1]))),
            'new_mae_pixels':float(np.mean(np.abs(pix[mask]-vy[mask,1]))),'median_true':float(np.median(vy[mask,1])),
            'median_old_estimate':float(np.median(vy[mask,5])),'median_new_estimate':float(np.median(pix[mask]))}
            for subset,mask in [('spatially_valid',usable),('old_missed_clear_valid',miss)]}
    report=dict(train=tra,calibration=val,threshold=THRESH,reference_old=old,refit=stats,model=cfg,
                house_crossfit_for_area_logit=3,calibration_candidates=summaries,
                meets_validation_error_constraint=bool(key[0]),calibration_pixel_accuracy=checks,
                previously_missed_clear_calibration_frames_recovered=int((miss&(newp*vy[:,4]>=THRESH)).sum()),
                per_calibration_house=[dict(old=classify(vy[m,3],vy[m],vg[m]),refit=classify(newp[m],vy[m],vg[m])) for house in np.unique(vh) for m in [vh==house]],
                weights_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())},
                sample_unit='Two motion sweeps per source cue. Alternate physically simulated oblique views and prior route samples. Foreground and pixel labels only in supervision/scoring; no held-out evaluation houses in fitting or selection.')
    (HERE/'clarity_training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
