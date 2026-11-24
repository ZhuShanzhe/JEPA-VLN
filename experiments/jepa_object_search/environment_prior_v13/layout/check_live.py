"""Causal replay: same scores from visited observations, regardless of other subtree states."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scene_layout import EpisodeLayout


def check(root):
    d=root/'layout_update';weights=json.loads((d/'frozen.json').read_text())
    checked=0;max_error=0.
    for h in sorted((root/'online/scored/fresh').iterdir()):
        f=next(iter(sorted(h.glob('*.npz'))));saved=d/'fresh'/h.name/f.name
        with np.load(f) as src,np.load(saved) as scores:
            paths=src['paths'].tolist();plans=src['plans'];bumps=src['blocked'];priors=src['prior']
            picks=[int(hashlib.sha256((f.stem+str(i)).encode()).hexdigest()[:6],16)%3 for i in range(4)]
            for mode in ('direct','jepa'):
                st=EpisodeLayout();current='r';block=src['env_pred_'+mode]
                for br in picks:
                    j=paths.index(current);base=block[:,j,0]
                    new=st.predict(base,plans[j],weights['weights'][mode],weights['strength'][mode])
                    delta=np.einsum('sak,sqk->sqa',new-base,priors[:,j])
                    err=float(np.max(abs(delta-scores['adjust_'+mode][:,j])))
                    if err>2e-6:raise ValueError('live score mismatch')
                    max_error=max(max_error,err);checked+=1
                    current+=str(br)
                    if len(current)<=4:
                        reached=src['sensed'][paths.index(current),:3]
                        st.observe(base[:,br],plans[j,br],bumps[j,br],reached)
    return dict(houses=12,all_cached_seeds_checked=True,comparisons_per_branch_across_queries=True,
                model_node_checks=checked,max_abs_error=max_error,
                inputs='only the chosen ancestor actions, collisions and three arrival observations')


if __name__=='__main__':
    a,b=map(Path,sys.argv[1:]);b.write_text(json.dumps(check(a),indent=2))
