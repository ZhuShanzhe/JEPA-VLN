"""Train foreground, cue trust, and independent translated-view reanchoring models on training buildings."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from threadpoolctl import threadpool_limits
from memory_refit import HERE,OLD,MODELS,Surface,FrameConfirmer,dense,model
from fit_confirm import pack,results
DATA=OLD/'private_confirm_data'
SEED=829

def sample_files(split):
    paths=sorted((DATA/split).glob('*/*.npz'))
    manifest=json.loads((HERE.parent/'new_view_20261129/view_design.json').read_text())['cohorts'][split]
    assert {p.parent.name for p in paths}==set(manifest)
    assert len(paths)==24*len(manifest)
    return paths

def variants(box,side):
    l,t,r,b=map(int,box);w,h=r-l,b-t
    yield (l,t,r,b)
    yield (max(0,l-w//2),max(0,t-h//2),min(192,r+w//2),min(192,b+h//2))
    if side:
        yield (max(0,l-w),max(0,t-h//3),r,min(192,b+h//4))
    else:
        yield (l,max(0,t-h//3),min(192,r+w),min(192,b+h//4))

def read(path):
    with np.load(path) as d:
        rgb,depth,pose,raw,box=[d[k] for k in ('rgb','depth','pose','mask','box')]
    masks=np.unpackbits(raw,axis=1).reshape(-1,192,192).astype(bool)
    return rgb,depth.astype(np.float32),pose,masks,box

def seed(path):
    return (int(path.stem)*29+sum(map(ord,path.parent.name))+SEED)%2147483648

def fit_pixels(paths):
    X=[];Y=[]
    for p in paths:
        rgb,dep,poses,masks,bb=read(p);rng=np.random.default_rng(seed(p))
        choices=[n for n in range(1,len(rgb)) if masks[n].sum()>=40]
        frames=[0]+choices[::max(1,len(choices)//3)][:3]
        for k in frames:
            y,x=np.where(masks[k]);box=(x.min(),y.min(),x.max()+1,y.max()+1)
            for bb in variants(box,seed(p)%2):
                f,u,v=dense(rgb[k],dep[k],bb)
                ix=rng.choice(len(f),min(100,len(f)),replace=False)
                X.extend(f[ix]); Y.extend(masks[k,v[ix],u[ix]])
    X=np.asarray(X);Y=np.asarray(Y)
    fit('foreground',X,Y)
    return dict(sampled_pixels=len(Y),background=int((~Y).sum()))

def fit(name,x,y):
    m=GradientBoostingClassifier(n_estimators=72,max_depth=3,min_samples_leaf=10,learning_rate=.065,random_state=SEED)
    m.fit(x,y);pack(m,MODELS/(name+'.npz'),x);model.cache_clear()

def extract(path):
    rgb,dep,poses,masks,orig=read(path);x=[];y=[];a=[];b=[];meta=[];pairmeta=[]
    for j,box in enumerate(variants(orig,seed(path)%2)):
        surf=Surface(rgb[0],dep[0],box);legacy=FrameConfirmer(rgb[0],dep[0],box)
        # Row-major indices match initial verifier deterministic sampling.
        l,t,r,bt=box
        yy,xx=np.mgrid[t:bt,l:r]
        stride=max(1,int(np.ceil(xx.size/6000)))
        fg=masks[0,yy.ravel()[::stride],xx.ravel()[::stride]][legacy.w>=.35]
        pix=masks[0,t:bt,l:r].sum();fill=pix/surf.area
        reliable=(len(fg)>=12 and fg.mean()>=.72 and not (pix<160 and fill<.35))
        x.append(surf.summary());y.append(bool(reliable));meta.append([j,int(pix),float(fill),len(fg),float(fg.mean()) if len(fg) else 0])
        for i,pose in enumerate(poses):
            if np.hypot(*pose[:2])<.12:continue
            candidate=surf.reproject(rgb[i],dep[i],pose)
            if candidate is None:continue
            feat,other=candidate;vs=masks[i,other.y,other.x][other.p>=.35]
            ok=(len(vs)>=12 and vs.mean()>=.72 and masks[i].sum()>=60 and results(masks[i],other.box)[1])
            a.append(feat);b.append(bool(ok));pairmeta.append([j,i])
    return x,y,a,b,meta,pairmeta

def assemble(paths):
    X=[];Y=[];A=[];B=[];M=[];G=[];I=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for f,result in enumerate(pool.map(extract,paths)):
            x,y,a,b,m,pm=result;X+=x;Y+=y;A+=a;B+=b;M+=m;G += [[f,j] for j in range(len(y))];I += [[f,*p] for p in pm]
    return [np.array(z) for z in (X,Y,A,B,M,G,I)]

def select(score,y,mask,min_precision):
    opts=[]
    for t in np.arange(.35,.995,.005):
        keep=(score>=t)&mask;n=int(keep.sum());g=int(y[keep].sum())
        if n>=15 and g/n>=min_precision:opts.append((g,-(n-g),float(t)))
    if opts:return max(opts)[2]
    return 1.01

def metrics(y,keep):
    n=int(keep.sum());g=int(y[keep].sum())
    return dict(n=len(y),passed=n,valid=g,invalid=n-g)

def train():
    os.umask(0o077);MODELS.mkdir(exist_ok=True)
    tr_files=sample_files('train');val_files=sample_files('val')
    px=fit_pixels(tr_files)
    tr=assemble(tr_files);va=assemble(val_files)
    for i,name in [(0,'initial'),(2,'relook')]:fit(name,tr[i],tr[i+1])
    x,y,a,b,m,g,inds=va
    p=model('initial')(x);q=model('relook')(a)
    cutoff=select(p,y,np.ones(len(y),bool),.94);passed=p>=cutoff
    # For each provisional source, calibrate the first promotion only.
    pending={tuple(k) for k in g[~passed]}
    rows=np.array([tuple(v[:2]) in pending for v in inds])
    def first(keep):
        ix=np.flatnonzero(keep); _,j=np.unique(inds[ix,:2],axis=0,return_index=True)
        mark=np.zeros(len(q),bool);mark[ix[j]]=True
        return mark
    options=[]
    for t in np.arange(.45,.995,.005):
        keep=rows&(q>=t);mark=first(keep);n=mark.sum();good=b[mark].sum()
        if n>=12 and good/n>=.95 and b[keep].mean()>=.95:options.append((int(good),-int(n-good),float(t)))
    second=max(options)[2] if options else 1.01
    (MODELS/'gates.json').write_text(json.dumps(dict(initial=cutoff,relook=second))+'\n')
    d=dict(train=dict(houses=12,cues=len(tr_files),**px,initial_boxes=len(tr[1]),initial_unreliable=int((~tr[1]).sum()),
                       relook_pairs=len(tr[3]),bad_associations=int((~tr[3]).sum()),
                       small_background_boxes=int(((tr[4][:,1]<160)&(tr[4][:,2]<.35)).sum())),
           calibration=dict(houses=6,cues=len(val_files),initial=metrics(y,passed),
               initial_native=metrics(y[m[:,0]==0],passed[m[:,0]==0]),
               small_background=metrics(y[(m[:,1]<160)&(m[:,2]<.35)],passed[(m[:,1]<160)&(m[:,2]<.35)]),
               first_reanchors=metrics(b,first(rows&(q>=second)))),
           thresholds=dict(initial=cutoff,relook=second,confirmation='unchanged'),
           association='First view uses independent RGB-D cue-foreground prediction. Pending memory requires real translation of >=0.12 m, geometrically consistent surfaces and freshly predicted destination foreground. Reanchor reference only after passing a trained/calibrated association check.',
           samples='Native, expanded and asymmetrically padded boxes around the same training instance, with source-pixel purity and same-instance second-view labels. 12/6 disjoint training/calibration houses. Inference uses only RGB-D and relative odometry.',
           hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(MODELS.iterdir())})
    (HERE/'memory_fit.json').write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps(d),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):train()
