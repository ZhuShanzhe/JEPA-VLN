"""Fit an anchor-conditioned foreground mask, then learn how mask evidence maps to a reportable footprint."""
import json
import os
from hashlib import sha256
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from area_verifier import HERE, REFINED,frame_features, Trees,combine
from fit_clarity import classify,THRESH

MODELS=HERE/'private'

def fitpixel(files):
    rng=np.random.RandomState(1199)
    xs=[];ys=[]
    for f in files:
        with np.load(f) as d:
            p,y,o=d['px'],d['gt'],d['offset']
            idx=np.concatenate([rng.choice(np.arange(a,b),size=min(64,b-a),replace=False) for a,b in zip(o,o[1:])])
            xs.append(p[idx]);ys.append(y[idx])
    net=HistGradientBoostingClassifier(max_iter=80,max_depth=4,max_leaf_nodes=15,min_samples_leaf=42,
                                      early_stopping=False,l2_regularization=1,random_state=1199)
    return net.fit(np.concatenate(xs),np.concatenate(ys))

def pack(net,path,sample,classification):
    allnodes=[p[0].nodes for p in net._predictors]
    nt=len(allnodes); n=max(map(len,allnodes))
    d={k:np.zeros((nt,n),dtype=float if k in ('threshold','value') else int) for k in ('key','left','right','value','threshold')}
    for i,row in enumerate(allnodes):
        for key,field in [('left','left'),('right','right'),('value','value'),('threshold','num_threshold')]:
            d[key][i,:len(row)]=row[field]
        d['key'][i,:len(row)]=np.where(row['is_leaf'],-1,row['feature_idx'].astype(int))
    np.savez_compressed(path,depth=net.max_depth,rate=1.,initial=float(net._baseline_prediction.ravel()[0]),**d)
    expected=net.predict_proba(sample)[:,1] if classification else 1/(1+np.exp(-net.predict(sample)))
    assert np.allclose(Trees(path)(sample),expected,rtol=1e-6,atol=1e-6)

def frame_data(files,pixel):
    frames=[];targets=[];groups=[];houses=[]
    for f in files:
        with np.load(f) as d:
            x=d['px'].astype('float32');p=pixel.predict_proba(x)[:,1]
            inside=d['interior'];off=d['offset'];orig=d['orig']
            for a,b,s,base in zip(off,off[1:],d['scale'],orig):
                sl=slice(a,b)
                frames.append(frame_features(p[sl],x[sl],inside[sl],s,base))
            targets.append(d['labels']);groups.extend([int(f.stem)]*len(orig));houses.extend([f.parent.name]*len(orig))
    return np.asarray(frames),np.concatenate(targets),np.asarray(groups),np.asarray(houses)

def make_head(cls):
    return cls(max_iter=100,max_depth=3,min_samples_leaf=18,early_stopping=False,
        l2_regularization=2,learning_rate=.075,random_state=1199)

def main():
    os.umask(0o077)
    train=sorted((MODELS/'features'/'train').glob('*/*.npz'))
    val=sorted((MODELS/'features'/'val').glob('*/*.npz'))
    homes={f.parent.name for f in train};vh={f.parent.name for f in val}
    assert len(homes)==12 and len(vh)==6 and not homes&vh and THRESH>=.835
    folds=np.array_split(sorted(homes),3)
    cross=[]
    for homes_out in folds:
        omitted=set(homes_out)
        pix=fitpixel([f for f in train if f.parent.name not in omitted])
        cross.append(frame_data([f for f in train if f.parent.name in omitted],pix))
    x,y,g,h=[np.concatenate(v) for v in zip(*cross)]
    pixel=fitpixel(train)
    with np.load(train[0]) as d:pack(pixel,MODELS/'fg.npz',d['px'][:100].astype('float32'),True)
    vx,vy,vg,_=frame_data(val,pixel)
    legacy=Trees(REFINED/'private_clarity_model'/'clear_frame.npz')
    assert json.loads((REFINED/'private_clarity_model'/'calibration.json').read_text())['area_weight']==0.
    prev=legacy(vx[:,20:])
    target=np.log((y[:,1]+1)/370)
    valid=(y[:,1]>=369)&(y[:,2]>0)
    weights=1.+.8*((y[:,1]>=152)&(y[:,1]<=820))
    # Calibrate the area-regression output using house-disjoint train predictions.
    oof=np.zeros(len(x))
    for ho in folds:
        ix=np.isin(h,ho)
        reg=make_head(HistGradientBoostingRegressor).fit(x[~ix],target[~ix],sample_weight=weights[~ix])
        oof[ix]=reg.predict(x[ix])
    lr=LogisticRegression(C=3).fit(oof[:,None],valid)
    area=make_head(HistGradientBoostingRegressor).fit(x,target,sample_weight=weights)
    cp=make_head(HistGradientBoostingClassifier).fit(x,valid,sample_weight=weights)
    pack(area,MODELS/'footprint.npz',vx[:200],False)
    pack(cp,MODELS/'clear.npz',vx[:200],True)
    vp=lr.predict_proba(area.predict(vx)[:,None])[:,1];vc=cp.predict_proba(vx)[:,1]
    base=classify(prev,vy,vg);best=None;choices=[]
    for fusion in ('mean',1.,1.25,1.5,2.):
      for old_w in (0.,.25,.5,.75,1.):
        if old_w==1 and fusion!='mean':continue
        for aw in (0.,.5,1.):
            score=combine(aw*vp+(1-aw)*vc,prev,old_w,fusion)
            m=classify(score,vy,vg)
            # Do not select higher first-report mistakes or a worse frame error fraction on calibration.
            okay=(m['first_bad']<=base['first_bad'] and m['bad_frames']/max(1,m['accepted_frames'])<=base['bad_frames']/max(1,base['accepted_frames']))
            key=(okay,m['first_correct'],-m['first_bad'],-m['bad_frames'], -old_w,aw)
            choices.append(dict(fusion=fusion,weight_old=old_w,weight_area=aw,eligible=okay,**m))
            if best is None or key>best[0]:best=(key,old_w,aw,m,fusion)
    _,old_w,aw,m,fusion=best
    cfg=dict(a=float(lr.coef_[0,0]),b=float(lr.intercept_[0]),area_weight=aw,old_weight=old_w,fusion=fusion)
    (MODELS/'cal.json').write_text(json.dumps(cfg)+'\n')
    report=dict(train_houses=12,calibration_houses=6,train_views=len(x),calibration_views=len(vx),frozen_product_threshold=THRESH,
        original_calibration=base,selected_calibration=m,held_house_pixel_folds=3,
        selection='First correct submissions maximized under no increase in first errors or bad accepted-frame fraction, validation houses only.',
        selected=cfg,options=choices,
        mean_absolute_pixel_error=float(np.abs(370*np.exp(area.predict(vx))-1-vy[:,1]).mean()),
        sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in [MODELS/n for n in ('cal.json','fg.npz','clear.npz','footprint.npz')]})
    (HERE/'area_training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(cfg,base,m,flush=True)
if __name__=='__main__':main()
