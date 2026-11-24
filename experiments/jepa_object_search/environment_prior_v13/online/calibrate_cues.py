"""Calibrate the frozen RGB target cue on true first appearances and preceding empty observations."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score


def samples(project, split):
    old=project/'jepa_hidden_v11'; prev=project/'jepa_route_v12'
    cache=(prev if split=='extra_train' else old)/'models/cache/hssd'/split
    out=[]
    for path in sorted((cache/'actor').glob('*.npz')):
        with np.load(path) as a:
            origin=str(a['origin']);house=str(a['house']);words=a['words']
        with np.load(prev/'data/local_targets'/split/(origin+'.npz')) as truth:
            first=truth['first_tap']
            assert np.array_equal(words,truth['words'])
        with np.load(cache/'teacher'/path.name) as obs:
            cue=obs['future_cue']
        for k in range(3):
            for q,t in enumerate(first[k]):
                end=t+1 if t>=0 else 4
                out.extend((house,float(cue[k,s,q]),int(s==t)) for s in range(end))
    return out


def main(project,out):
    train=samples(project,'train')+samples(project,'extra_train')
    cal=samples(project,'calibration')
    h,x,y=map(np.array,zip(*train))
    weight=np.zeros(len(x))
    for a in np.unique(h):
        weight[h==a]=1./sum(h==a)
    # Pool adjacent values into bins before fitting to keep small tails from dominating.
    grid=np.unique(np.quantile(x,np.linspace(0,1,25)))
    cell=np.minimum(np.searchsorted(grid[1:],x),len(grid)-2)
    xs,ys,ws=[],[],[]
    for i in np.unique(cell):
        take=cell==i
        xs.append(np.average(x[take],weights=weight[take]))
        ys.append(np.average(y[take],weights=weight[take]))
        ws.append(weight[take].sum())
    model=IsotonicRegression(out_of_bounds='clip').fit(xs,ys,sample_weight=ws)
    _,xt,yt=map(np.array,zip(*cal));hat=model.predict(xt)
    doc=dict(raw=model.X_thresholds_.tolist(),evidence=model.y_thresholds_.tolist(),
             training_frames=len(train),training_houses=len(np.unique(h)),calibration_frames=len(cal),
             train_first_appearance_frequency=float(np.average(y,weights=weight)),
             calibration_auc=float(roc_auc_score(yt,hat)),calibration_brier=float(np.mean((hat-yt)**2)))
    out.write_text(json.dumps(doc,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('project',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();main(a.project,a.output)
