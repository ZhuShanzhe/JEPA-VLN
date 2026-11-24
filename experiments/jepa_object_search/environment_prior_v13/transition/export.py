"""Export calibrated room forecasts for every candidate, using only departure views."""
import argparse
from pathlib import Path
import sys
import numpy as np
from transition import TransitionPredictor


def run(project,split):
    d=project/'jepa_env_v13'
    sys.path.insert(0,str(d/'models'))
    from env_routes import EnvironmentRoutes
    heads=[TransitionPredictor(project,EnvironmentRoutes(project,s),s,residual=True) for s in (11,29,47)]
    if not heads[0].mix:return
    for src in sorted((d/'online/visual'/split).glob('*/*.npz')):
        dst=d/'transition'/split/src.parent.name/src.name
        with np.load(src) as a:
            # The first three frames are captured at the junction before motion.
            contexts=a['features'][:,:3];plans=a['plans'];goals=a['goals'];paths=a['paths']
        pred=np.stack([[m.forecast(x,h).mean(axis=1) for x,h in zip(contexts,plans)] for m in heads])
        values=np.repeat(pred[:,:,None],len(goals),axis=2)
        dst.parent.mkdir(parents=True,exist_ok=True)
        tmp=dst.with_suffix('.partial.npz')
        np.savez_compressed(tmp,paths=paths,goals=goals,env_pred_jepa=values)
        tmp.replace(dst)
    print('prepared',split,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('project',type=Path);p.add_argument('split',choices=['fresh','calibration'])
    a=p.parse_args();run(a.project,a.split)
