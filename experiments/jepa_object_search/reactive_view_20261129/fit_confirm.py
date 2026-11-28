"""Fit foreground occupancy, track agreement and visible-frame models; calibrate on separate buildings."""
from pathlib import Path
from collections import Counter
import hashlib,json,os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from frame_confirm import HERE,MODELS,FrameConfirmer,source_features,load
from threadpoolctl import threadpool_limits

def pack(model,path,x):
    trees=[e[0].tree_ for e in model.estimators_]
    n=max(t.node_count for t in trees)
    arrays={k:np.zeros((len(trees),n),dtype=float if k in ("threshold","value") else int) for k in ("key","threshold","value","left","right")}
    for i,t in enumerate(trees):
        for k,v in (("key",t.feature),("threshold",t.threshold),("left",t.children_left),("right",t.children_right),("value",t.value[:,0,0])):
            arrays[k][i,:t.node_count]=v
    freq=model.init_.class_prior_
    np.savez_compressed(path,depth=model.max_depth,rate=model.learning_rate,initial=np.log(freq[1]/freq[0]),**arrays)
    from frame_confirm import Trees
    assert np.allclose(Trees(path)(x[:150]),model.predict_proba(x[:150])[:,1],atol=1e-6)

def bounds(mask):
    y,x=np.where(mask)
    if len(y)==0:return None
    return [x.min(),y.min(),x.max()+1,y.max()+1]

def results(mask,box):
    n=int(mask.sum());l,t,r,b=box
    inside=int(mask[t:b,l:r].sum())
    true=bounds(mask)
    if true is None:return False,False
    L,T,R,B=true
    intersection=max(0,min(R,r)-max(L,l))*max(0,min(B,b)-max(T,t))
    union=(R-L)*(B-T)+(r-l)*(b-t)-intersection
    return n>=369, inside>=.5*n and intersection/max(union,1)>=.3

def samples(split):
    files=sorted((HERE/"private_confirm_data"/split).glob("*/*.npz"))
    assert len(files)==(288 if split=="train" else 144)
    return files

def fit_source():
    rng=np.random.default_rng(911)
    xx=[];yy=[]
    for f in samples("train"):
        with np.load(f) as d:
            rgb=d["rgb"][0];dep=d["depth"][0];mask=np.unpackbits(d["mask"][0]).reshape(192,192);box=d["box"]
        x,u,v=source_features(rgb,dep,box)
        ix=rng.choice(len(x),min(240,len(x)),replace=False)
        xx.append(x[ix]);yy.append(mask[v[ix],u[ix]])
    x=np.concatenate(xx);y=np.concatenate(yy)
    m=GradientBoostingClassifier(n_estimators=48,max_depth=3,learning_rate=.09,random_state=911)
    m.fit(x,y);pack(m,MODELS/"source.npz",x)
    return dict(sampled_source_pixels=len(y),non_target_pixels=int((y==0).sum()))

def assemble(split):
    features=[];cl=[];sm=[];groups=[];frames=[];total=0;opps=set()
    for group,f in enumerate(samples(split)):
        with np.load(f) as d:
            rgb,dep,pose,mask,box=(d[k] for k in ("rgb","depth","pose","mask","box"))
        tracker=FrameConfirmer(rgb[0],dep[0],box)
        for i,p in enumerate(pose):
            true=np.unpackbits(mask[i]).reshape(192,192)
            total+=1
            if true.sum()>=369:opps.add(group)
            x,b=tracker.evidence(rgb[i],dep[i],p)
            if b is None:continue
            c,s=results(true,b)
            features.append(x);cl.append(c);sm.append(s);groups.append(group);frames.append(i)
    return np.array(features),np.array(cl),np.array(sm),np.array(groups),np.array(frames),dict(frames=total,groups_with_true_opportunity=len(opps))

def first(groups,score,good,threshold):
    ix=np.flatnonzero(score>=threshold)
    _,j=np.unique(groups[ix],return_index=True); chosen=ix[j]
    return len(chosen),int(good[chosen].sum())

def main():
    os.umask(0o077);MODELS.mkdir(exist_ok=True)
    report=dict(source=fit_source())
    train=assemble("train");val=assemble("val")
    x,c,s,g,idx,_=train;vx,vc,vs,vg,vi,_=val
    for name,y in (("clear",c),("same",s)):
        model=GradientBoostingClassifier(n_estimators=54,max_depth=3,learning_rate=.075,min_samples_leaf=14,random_state=911)
        model.fit(x,y);pack(model,MODELS/(name+".npz"),x)
    score=load("clear")(vx)*load("same")(vx);good=vc&vs
    options=[]
    for t in np.linspace(.05,.995,190):
        n,yes=first(vg,score,good,t)
        frame=(score>=t)
        if n>=12 and yes/n>=.95 and good[frame].mean()>=.95: options.append((yes,-(n-yes),t))
    threshold=max(options)[2] if options else 1.1
    (MODELS/"thresholds.json").write_text(json.dumps(dict(threshold=threshold))+"\n")
    n,yes=first(vg,score,good,threshold);admit=score>=threshold
    report.update(train=dict(train[-1],supported_frames=len(c)),val=dict(val[-1],supported_frames=len(vc),groups=144,
       frame_accept=int(admit.sum()),frame_valid=int(good[admit].sum()),first_accept=n,first_valid=yes,
       accepted_but_unclear=int((admit&~vc).sum()),accepted_but_bad_identity_box=int((admit&~vs).sum())),
       calibration="Require at least 95% precision for frames and first accepted frames per cue across the enumerated calibration plans; maximize true first accepts among threshold candidates. No semantic inputs during inference.",
       same_target="Prediction rectangle includes >=50% visible pixels of original instance with box IoU>=0.3. At least 369 visible pixels for clear. No oracle mask in source or during decisions.",
       threshold=float(threshold),model_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(MODELS.glob("*"))})
    (HERE/"frame_confirm_fit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)

if __name__=="__main__":
    with threadpool_limits(limits=2):main()
