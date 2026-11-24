"""Cross-check actual observed junction RGB and prefix against a frozen delta bank."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from train_value import ValueResidual


def run(project):
    r=project/'jepa_env_v13';sys.path.insert(0,str(r/'models'))
    from env_routes import EnvironmentRoutes
    model=EnvironmentRoutes(project,11)
    heads={a:ValueResidual.load(project,a) for a in ('jepa','direct')}
    errors=[];choices=[];stored_errors=[]
    for house in sorted((r/'simulator/trees/calibration').iterdir()):
        m=sorted(house.glob('*/episode.json'))[0];w=json.loads(m.read_text())['targets'][0]['word']
        origin=m.parent.name
        with np.load(r/'horizon/residual/calibration'/house.name/(origin+'.npz')) as saved, np.load(
                r/'online/scored/calibration'/house.name/(origin+'.npz')) as encoded:
            qi=saved['goals'].tolist().index(w)
            contexts=[];histcue=[]
            for p in ('r','r0','r00','r000'):
                with np.load(m.parent/('node_'+p+'.npz')) as z:
                    views=z['ctx'];plans=z['plans']
                sensed,cue=model.observe_batch(views,w)
                start=model.live(views,w,plans)
                at=saved['paths'].tolist().index(p)
                for a,head in heads.items():
                    d=head.route_delta(start[a],start['env_pred_'+a],start['prior'],sensed,cue,plans,5-len(p),
                        np.mean(contexts,axis=(0,1)) if contexts else sensed.mean(0),
                        float(np.mean(histcue)) if histcue else float(cue.mean()))
                    key='residual' if a=='jepa' else 'residual_direct'
                    cached=saved[key][0,at,qi]
                    pre=[encoded['paths'].tolist().index(p[:n]) for n in range(1,len(p))]
                    sc=encoded['sensed']; cu=encoded['cue']
                    exact=head.route_delta(encoded[a][0,at,qi],encoded['env_pred_'+a][0,at,qi],
                        encoded['prior'][0,at,qi],sc[at,:3],cu[at,:3,qi],plans,5-len(p),
                        sc[pre,:3].mean((0,1)) if pre else sc[at,:3].mean(0),
                        cu[pre,:3,qi].mean() if pre else cu[at,:3,qi].mean())
                    stored_errors.append(float(np.max(abs(exact-cached))))
                    errors.append(float(np.max(abs(cached-d))))
                    choices.append(int(np.argmax(start[a]+d)==np.argmax(start[a]+cached)))
                contexts.append(sensed);histcue.append(cue)
    # Visual descriptors are cached in float16; fresh encodings can have been
    # extracted in a different batch. Audit discrete choices and report both errors.
    if not all(choices) or max(errors)>np.finfo(np.float16).eps or max(stored_errors)>1e-6:
        raise ValueError('live and bank deltas disagree')
    return dict(comparisons=len(errors),matched_choices=sum(choices),max_raw_rgb_delta_error=max(errors),
                max_stored_input_delta_error=max(stored_errors))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('project',type=Path);opt=a.parse_args()
    (opt.project/'jepa_env_v13/horizon/live_check.json').write_text(json.dumps(run(opt.project),indent=2))
