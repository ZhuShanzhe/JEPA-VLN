"""Learn target occupancy for each official room type using training houses only."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys
import numpy as np
from rooms import Atlas,ROOMS,GROUPS


def source(project):
    base=project/'jepa_hidden_v11/hssd'
    more=project/'jepa_route_v12/expansion/hssd'
    old=json.loads((base/'cohort.json').read_text())
    new=json.loads((more/'cohort.json').read_text())
    splits={'train':(base/'train',old['train']),
            'extra_train':(more/'train',new['train']),
            'calibration':(base/'calibration',old['calibration'])}
    assert len(set(sum([v[1] for v in splits.values()],[])))==46
    return splits


def stats(assets,house,words,lookup):
    atlas=Atlas(assets,house)
    data=json.loads((assets/'scenes'/(house+'.scene_instance.json')).read_text())['object_instances']
    xyz=[obj['translation'] for obj in data]
    indices=atlas.locate(xyz,instance=True)
    positive=np.zeros((len(atlas.names),len(words)),int)
    assigned=unassigned=0
    for obj,region in zip(data,indices):
        ids=re.findall(r'[a-fA-F0-9]{40}',obj['template_name'])
        name=next((lookup[v] for v in ids if v in lookup),None)
        if name not in words:
            continue
        if region<0:
            unassigned+=1
            continue
        positive[region,words.index(name)]=1
        assigned+=1
    bins=np.bincount(atlas.names,minlength=len(ROOMS))
    counts=np.zeros((len(ROOMS),len(words)),int)
    for rid,row in zip(atlas.names,positive):
        counts[rid]+=row
    return bins,counts,assigned,unassigned


def main(project,assets,dst):
    sys.path.insert(0,str(project/'jepa_route_v12/hssd'))
    # Only the word taxonomy is read from the old collector.
    import ast
    module=ast.parse((project/'jepa_route_v12/hssd/collect_branches.py').read_text())
    words=list(next(ast.literal_eval(a.value) for a in module.body if isinstance(a,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='WORDS' for t in a.targets)))
    with (assets/'metadata/fpmodels-with-decomposed.csv').open() as f:
        lookup={r['id']:r['main_wnsynsetkey'] or r['wnsynsetkey'] for r in csv.DictReader(f)}
    houses=sum([v[1] for k,v in source(project).items() if k in ('train','extra_train')],[])
    parts=[stats(assets,h,words,lookup) for h in houses]
    total=np.sum([p[0] for p in parts],axis=0)
    pos=np.sum([p[1] for p in parts],axis=0)
    base=(pos.sum(0)+.5)/(total.sum()+1.)
    # Four pseudo-rooms with the training-set target prevalence.
    rate=(pos+4.*base)/(total[:,None]+4.)
    p=total/total.sum(); posterior=rate*p[:,None];posterior/=posterior.sum(axis=0,keepdims=True)
    record=dict(room_names=ROOMS,region_groups=GROUPS,p_room_prior=p.tolist(),
                p_room_given_target=dict(zip(words,posterior.T.tolist())),
                p_target_given_room=dict(zip(words,rate.T.tolist())),
                room_counts=total.tolist(),rooms_with_target=dict(zip(words,pos.T.tolist())),
                training_houses=len(houses),assigned_target_anchors=int(sum(p[2] for p in parts)),
                unassigned_target_anchors=int(sum(p[3] for p in parts)),
                sampling_unit='annotated physical room; each object category counted once per room',
                prior_pseudo_rooms=4)
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(record,indent=2))
    print('prior',len(houses),int(total.sum()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('project','assets','out'):
        p.add_argument(name,type=Path)
    a=p.parse_args();main(a.project,a.assets,a.out)
