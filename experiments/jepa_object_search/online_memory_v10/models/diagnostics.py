"""Attribute route changes by motion history. Uses the same frozen scores and no re-fitting."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from fit_score import D,FEATURES,H,read,design
from v10_live import prior_transform

def audit(split):
    weights=np.array([json.loads((D/'v10.json').read_text())['weights'][k] for k in FEATURES])
    pweight,gweight,nweight,hweight,tweight=weights
    by=defaultdict(lambda:defaultdict(list));den=defaultdict(int);counter=defaultdict(int)
    allhouse=defaultdict(lambda:defaultdict(list));magnitude=defaultdict(list)
    for house,sid,z in read(split):
        motion='translated' if z['translated'] else 'rotation_or_blockage'
        den[motion+'_origins']+=1
        for i,h in enumerate(H):
            f=z['f'][i];target=z['y'][i];p,g,nov,haz,tr=np.moveaxis(design(f),-1,0)
            fixed=p*pweight+g*gweight
            explore=fixed+nweight*nov
            spatial=explore+hweight*haz
            full=spatial+tweight*tr
            sc={'fixed':fixed,'exploration_added':explore,'exploration_and_obstacle':spatial,'full':full,
                'v8':f[:,:,0],'observed_memory':spatial-pweight*p+prior_transform(f[:,:,1])*pweight}
            qidx=np.flatnonzero(z['present'])
            for q in qidx:
                yq=target[:,q];choices={k:int(val[:,q].argmax()) for k,val in sc.items()}
                for k,idx in choices.items():allhouse[house][k].append(float(yq[idx]))
                if yq.max()==yq.min():continue
                den[motion+'_informative']+=1
                for k,idx in choices.items():by[motion][k].append(float(yq[idx]))
                for a,b in [('fixed','exploration_added'),('exploration_added','exploration_and_obstacle'),('exploration_and_obstacle','full'),('fixed','full')]:
                    name=a+'_to_'+b
                    if not np.array_equal(z['actions'][choices[a],:h],z['actions'][choices[b],:h]):
                        counter[name+'_changes']+=1
                        diff=float(yq[choices[b]])-float(yq[choices[a]])
                        counter[name+('_improve' if diff>0 else '_regress' if diff<0 else '_same')]+=1
                for k,val in {'prior':p*pweight,'geometry':g*gweight,'novelty':nov*nweight,'clearance':haz*hweight,'trace':tr*tweight}.items():
                    magnitude[k].append(float(val[:,q].max()-val[:,q].min()))
    means=lambda a:{k:float(np.mean(v)) for k,v in a.items()}
    unconditional={k:float(np.mean([np.mean(v[k]) for v in allhouse.values()])) for k in next(iter(allhouse.values()))}
    return dict(coverage=den,changes_pooled_counts=counter,informative_mean_gain_pooled_by_history={m:means(g) for m,g in by.items()},
                mean_absolute_candidate_range=means(magnitude),all_present_house_macro_top_gain=unconditional)
if __name__=='__main__':
    (D/'hssd_diagnostics.json').write_text(json.dumps({s:audit(s) for s in ('train','calibration','holdout')},indent=2))
