"""Train a second, foreground-verified proposal for translating to a new reference frame."""
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
from train_memory import variants,seed,results,sample_files,read,fit
from memory_refit import HERE,MODELS,Surface,FrameConfirmer,model,corroborate

def extract(path):
    rgb,dep,poses,masks,box=read(path);X=[];Y=[];G=[]
    for j,b in enumerate(variants(box,seed(path)%2)):
        src=Surface(rgb[0],dep[0],b);legacy=FrameConfirmer(rgb[0],dep[0],b)
        for i,p in enumerate(poses):
            if np.hypot(*p[:2])<.12:continue
            e,bb=legacy.evidence(rgb[i],dep[i],p)
            rec=corroborate(src,bb,e,rgb[i],dep[i],p)
            if rec is None:continue
            feat,view=rec;fg=masks[i,view.y,view.x][view.p>=.35]
            label=len(fg)>=8 and fg.mean()>=.6 and results(masks[i],bb)[1] and masks[i].sum()>=50
            X.append(feat);Y.append(label);G.append(j)
    return X,Y,G

def assemble(paths):
    X=[];Y=[];G=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i,(x,y,g) in enumerate(pool.map(extract,paths)):X+=x;Y+=y;G+=list(i*3+np.array(g,dtype=int))
    return np.asarray(X),np.asarray(Y),np.asarray(G)
def main():
    x,y,_=assemble(sample_files('train'));vx,vy,vg=assemble(sample_files('val'))
    fit('corroboration',x,y);prob=model('corroboration')(vx)
    opts=[]
    for t in np.arange(.3,.995,.005):
        ix=np.flatnonzero(prob>=t);_,j=np.unique(vg[ix],return_index=True)
        g=vy[ix[j]]
        if len(g)>=12 and g.mean()>=.95 and vy[ix].mean()>=.95:opts.append((int(g.sum()),-int((~g).sum()),float(t)))
    thresh=max(opts)[2] if opts else 1.01
    keys=json.loads((MODELS/'gates.json').read_text());keys['corroboration']=thresh
    (MODELS/'gates.partial').write_text(json.dumps(keys)+'\n');(MODELS/'gates.partial').replace(MODELS/'gates.json')
    d=dict(training_pairs=len(y),training_good=int(y.sum()),validation_pairs=len(vy),threshold=thresh,
            available_first_positive_and_negative=(list(max(opts)[:2]) if opts else []),
            second_reference='Require a displaced view, independent foreground agreement and cross-view geometric association. Retain first reference as secondary only when the two references agree on the current box at IoU >=0.55.')
    (HERE/'corroboration_fit.json').write_text(json.dumps(d,indent=2)+'\n');print(d,flush=True)
if __name__=='__main__':main()
