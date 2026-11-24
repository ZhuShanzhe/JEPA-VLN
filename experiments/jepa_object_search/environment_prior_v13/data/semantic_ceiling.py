"""Calibration-only oracle-room diagnostic. The route policy never receives this input."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from train_prior import source


def run(project,dst):
    d=project/'jepa_env_v13/data'
    prior=json.loads((d/'prior.json').read_text())
    base,houses=source(project)['calibration'];means=[]
    for house in houses:
        origins=[]
        for p in sorted((base/house).glob('*.npz')):
            with np.load(p) as z:words=z['words'].tolist();gain=z['y']>0
            with np.load(d/'labels/calibration'/p.name) as z: view=z['view_room'].copy();pos=z['room_idx']
            for i,t in np.argwhere(view.sum(-1)==0):
                if pos[i,t]>=0:view[i,t,pos[i,t]]=1.
                else:view[i,t]=prior['p_room_prior']
            queries=[]
            for q,word in enumerate(words):
                v=np.array(prior['p_target_given_room'][word]);rank=view.mean(1)@v
                # Expected outcome when sharing equal highest scores.
                best=np.isclose(rank,rank.max(),rtol=1e-7,atol=1e-9)
                queries.append([gain[:,q][best].mean(),gain[:,q].mean()])
            origins.append(np.mean(queries,axis=0))
        means.append(np.mean(origins,axis=0))
    result=np.mean(means,axis=0)
    dst.write_text(json.dumps(dict(houses=len(means),oracle_true_environment_plus_empirical_prior=float(result[0]),
                        uniform=float(result[1]),house_delta=[float(v[0]-v[1]) for v in means]),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('project',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.project,a.output)
