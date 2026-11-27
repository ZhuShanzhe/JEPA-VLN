"""Paired, house-level uncertainty for the frozen future-view and direct models."""
from collections import defaultdict
import json
import numpy as np
import torch
from train_views import HERE,SEED,arrays,assemble,inputs
from view_jepa import ViewPredictor


def main():
    torch.set_num_threads(4);torch.manual_seed(SEED)
    rows,x,e,_=inputs()
    sets,z=assemble(rows,x,e); d=arrays(sets['test'],x,z,rows)
    preds={k:ViewPredictor(HERE/'weights'/(k+'.npz')).forward(d['x'])[0]
           for k in ('direct','view_jepa')}
    groups=defaultdict(list); byhouse=defaultdict(list)
    for i,key in enumerate(d['anchor']):groups[key].append(i)
    output={}
    for subset in ('all','clock'):
        blocks=defaultdict(list)
        for ids in groups.values():
            i=ids[0]
            if len(ids)<2 or (subset=='clock' and not d['clock'][i]):continue
            y=d['y'][ids]
            quality=[y[np.isclose(p[ids],p[ids].max(),rtol=0,atol=1e-7)].mean() for p in preds.values()]
            blocks[str(d['house'][i])].append([*quality,y.mean()])
        vals=np.stack([np.mean(v,axis=0) for v in blocks.values()])
        diff=vals[:,1]-vals[:,0]
        boot=np.random.default_rng(SEED).choice(diff,(10000,len(diff)),replace=True).mean(1)
        output[subset]=dict(houses=len(vals),choices=sum(map(len,blocks.values())),
             house_macro_direct=float(vals[:,0].mean()),house_macro_jepa=float(vals[:,1].mean()),
             house_macro_random=float(vals[:,2].mean()),house_macro_difference=float(diff.mean()),
             paired_house_percentile_95=np.quantile(boot,[.025,.975]).tolist())
    (HERE/'heldout_views.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output),flush=True)


if __name__=='__main__':main()
