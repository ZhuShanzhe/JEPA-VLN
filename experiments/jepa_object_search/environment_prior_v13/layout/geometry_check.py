"""Compare endpoint predictions with independently reconstructed HSSD view-region labels."""
from pathlib import Path
from collections import defaultdict
import sys
import json
import numpy as np
from scene_layout import EpisodeLayout


def region(views):
    labels=np.asarray(views)[:3]
    good=labels.sum(-1)>0
    return labels[good].mean(0) if good.any() else None


def run(root):
    d=root/'layout_update';locked=json.loads((d/'frozen.json').read_text())
    acc=defaultdict(lambda:defaultdict(list))
    for f in sorted((root/'data/diagnostic_private').glob('*.json')):
        p=json.loads(f.read_text());h=f.stem;o=p['origin'];nodes=p['nodes']
        with np.load(root/'online/scored/fresh'/h/(o+'.npz')) as src:
            paths=src['paths'].tolist();plans=src['plans'];bumps=src['blocked'];sensed=src['sensed'][:,:3]
            for mode in ('jepa','direct'):
                bank=src['env_pred_'+mode];memory=EpisodeLayout()
                for t,(r,nxt) in enumerate(zip(nodes,nodes[1:])):
                    i=paths.index(r['node']);j=paths.index(nxt['node']);a=r['choice']
                    assert r['node']+str(a)==nxt['node']
                    base=bank[:,i,0]
                    corrected=memory.predict(base,plans[i],locked['weights'][mode],locked['strength'][mode])
                    current=region(r['labels']); target=region(nxt['labels'])
                    if t>0 and target is not None:
                        labels=['all']
                        if current is not None:labels+=['changed' if current.argmax()!=target.argmax() else 'same']
                        p0=base[:,a];p1=corrected[:,a]
                        scores=[np.mean(np.sum((z-target)**2,-1)) for z in (p0,p1)]
                        scores.extend(np.mean(z.argmax(-1)==target.argmax()) for z in (p0,p1))
                        for label in labels:acc[mode+'|'+label][h].append(scores)
                    # Only sensed RGB-derived probabilities go into the memory.
                    memory.observe(base[:,a],plans[i,a],bumps[i,a],sensed[j])
    out={}
    for group,hh in acc.items():
        vals=np.stack([np.mean(v,0) for v in hh.values()]);z=vals.mean(0)
        d=vals[:,1]-vals[:,0]
        interval=np.quantile(np.random.default_rng(131).choice(d,(12000,len(d))).mean(1),[.025,.975])
        out[group]=dict(houses=len(hh),transitions=sum(map(len,hh.values())),
                        original_brier=z[0],updated_brier=z[1],original_accuracy=z[2],updated_accuracy=z[3],
                        delta_brier_ci95=interval.tolist())
    return out


if __name__=='__main__':Path(sys.argv[2]).write_text(json.dumps(run(Path(sys.argv[1])),indent=2))
