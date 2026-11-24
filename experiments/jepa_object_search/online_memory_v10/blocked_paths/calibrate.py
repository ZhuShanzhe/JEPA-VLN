"""Estimate passability of a proposed forward budget on a revisited route using RGB only."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D=Path(__file__).resolve().parent
BUDGET=(4,12)
SPLITS={'train':range(8),'validation':range(8,10),'test':range(10,14)}
def region_stats(a):
    return [float(x) for patch in np.array_split(a,3,axis=0) for cell in np.array_split(patch,3,axis=1)
            for x in [np.mean(cell),np.quantile(cell,.75)]]
def gray(im):
    return cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype('float32')/255.
def edge(im):
    g=gray(im)
    return np.clip(np.abs(cv2.Sobel(g,-1,1,0))+np.abs(cv2.Sobel(g,-1,0,1)),0,1)
def current(rgb,h):
    # Centre-relative gradients and local brightness contrast; no object ID or scene key.
    g=gray(rgb);e=edge(rgb)
    shape=region_stats(np.abs(g-np.median(g)))+region_stats(e)
    return [h/12.,*shape,*[x*h/12. for x in shape]]
def change(before,after,h):
    a=before.astype('float32')/255.;b=after.astype('float32')/255.
    dif=np.abs(a-b).mean(-1);ed=np.abs(edge(before)-edge(after))
    # A moving side distractor is present for both gate events and stable gates.
    shape=region_stats(dif)+region_stats(ed)
    return [h/12.,*shape,*[x*h/12. for x in shape]]
def model():return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=600,random_state=923))
def score(prior_view,current_view,motion,h,current_model,change_model):
    # motion is the measured RGB change along the past executed actions, with no post-event contact.
    old=float(np.asarray(motion[:h],bool).sum()>=h-.5)
    vv=current_model.predict_proba([current(current_view,h)])[0,1]
    transition=change_model.predict_proba([change(prior_view,current_view,h)])[0,1]
    changed=transition*(1-old)+(1-transition)*old
    return dict(current_rgb=float(vv),collision_memory=old,updated=float(changed))
def rows(split):
    arr=[]
    for i in SPLITS[split]:
        name=f'h{i:02d}'
        for initial in range(2):
            for swapped in range(2):
                uid=f'{name}_{initial}_{swapped}'
                with np.load(D/'records_v3'/(uid+'.obs.npz')) as z:a,b,m=z['before'],z['after'],z['prior_movement']
                eco=json.loads((D/'records_v3'/(uid+'.truth.json')).read_text());truth=eco['outcomes'][1]
                for side in range(2):
                    for h in BUDGET:
                        # Physical target and passability labels are attached outside feature functions.
                        yy=bool(truth[side]['d'+str(h)]>=.25*h-.16);old=bool(sum(m[side,:h])>=h-.5)
                        arr.append(dict(name=name,id=uid,h=h,side=side,swapped=swapped,old=old,y=yy,
                            near=truth[side]['near'],dlong=truth[side]['d12'],initial_open=eco['outcomes'][0][side]['d12']>=2.84,
                            rgb=a[side],now=b[side],moves=m[side],event=eco['neutral_event_rgb_mae'],
                            drift=max(xx['drift'] for phase in eco['outcomes'] for xx in phase)))
    return arr
def train():
    data=rows('train')
    vision=model().fit([current(x['now'],x['h']) for x in data],[x['y'] for x in data])
    update=model().fit([change(x['rgb'],x['now'],x['h']) for x in data],[x['y']!=x['old'] for x in data])
    freq={h:float(np.mean([r['y'] for r in data if r['h']==h])) for h in BUDGET}
    avg=lambda a,b:float(np.abs(a.astype('float32')-b).mean()/255)
    longs=[r for r in data if r['h']==12];values=np.array([avg(r['rgb'],r['now']) for r in longs]);labels=np.array([r['old']!=r['y'] for r in longs])
    thr=min(np.linspace(0,1,251),key=lambda t:float(np.mean((values>t)!=labels)))
    path=D/'passability.joblib'
    joblib.dump(dict(visual=vision,update=update,frequency=freq,raw_threshold=thr),path)
    return joblib.load(path)
def summary(data,ck):
    arms=['fixed_frequency','collision_memory','current_rgb','updated','raw_rgb_change','blanket_dead_end_veto']
    parts=defaultdict(lambda:defaultdict(list));episodes=defaultdict(lambda:defaultdict(list))
    val=lambda d:{k:float(np.mean(x)) for k,x in d.items()}
    for r in data:
        h=r['h'];s=score(r['rgb'],r['now'],r['moves'],h,ck['visual'],ck['update'])
        diff=float(np.abs(r['rgb'].astype('float32')-r['now']).mean()/255.)
        s['fixed_frequency']=ck['frequency'][h];s['raw_rgb_change']=float(bool(r['old']) ^ bool(h==12 and diff>ck['raw_threshold']))
        # Conventional unconditional rejection: a collision anywhere in the earlier 3m probe vetoes the entire side.
        s['blanket_dead_end_veto']=float(np.asarray(r['moves']).sum()>=11.5)
        tag=('reopened' if r['y'] else 'newly_blocked') if r['swapped'] else 'stable'
        for arm,p in s.items():
            correct=float((p>.5)==r['y']) if abs(p-.5)>1e-9 else .5
            if h==12:
                for t in ('long_all',tag):parts[t][arm].append(correct)
                episodes[r['id']][arm].append([p,r['y']])
                parts['long_brier'][arm].append((p-r['y'])**2)
            else:
                parts['short_all'][arm].append(correct)
                if r['near'] and r['dlong']<2.84 and r['y']:
                    parts['short_reachable_target_in_closed_lane'][arm].append(float(p>=.5))
    choices=defaultdict(list)
    for uid,d in episodes.items():
        for arm,v in d.items():
            p=np.asarray([a for a,y in v]);y=np.asarray([y for a,y in v],float)
            if y.sum()!=1:raise ValueError('expected one open long route')
            choices[arm].append(float(np.mean(y[p==p.max()])))
    return dict(layouts=sorted({x['name'] for x in data}),episodes=len(episodes),decision_records=len(data),counts={k:len(next(iter(v.values()))) for k,v in parts.items()},
                scores={k:val(v) for k,v in parts.items()},long_route_selection=val(choices),
                max_return_drift_m=max(x['drift'] for x in data),max_neutral_view_change=max(x['event'] for x in data))
if __name__=='__main__':
    ck=train()
    # Splits, .5 decision threshold and model capacity are fixed before evaluation.
    result=dict(task='physical budget-conditioned 1m / 3m route reachability after observing a revisited corridor; simulated returns and gate relocation in same world',
        models='prior, actual historical RGB-stall memory, current-only visual classifier, time-adaptive change classifier conditioned on history and plan budget, raw image difference and unconditional whole-side veto',
        split_geometry='Geometry, materials, barrier lengths and widths are seeded per unique layout. 8 train / 2 validation / 4 held-out layouts. Static and swapped gate controls both include an unrelated moving distractor.',
        weights_sha256=hashlib.sha256((D/'passability.joblib').read_bytes()).hexdigest(),sklearn_version=__import__('sklearn').__version__,
        splits={s:summary(rows(s),ck) for s in SPLITS})
    (D/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('DONE',result['splits']['test'],flush=True)
