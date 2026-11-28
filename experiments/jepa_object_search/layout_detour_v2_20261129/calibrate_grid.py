"""Calibrate route ranking on GT-verified endpoints from 6 calibration houses, known-free routes only."""
import json, sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
V1=HERE.parent/'layout_detour_20261129'
sys.path.insert(0,str(V1))
from fit_layout import load,BASE
from layout_model import learned_net
def main():
    split=json.loads((BASE/'view_design.json').read_text())['cohorts']
    v,y,_,_=load('val',split['val'])
    ckpt=torch.load(V1/'private_layout.pt',weights_only=True,map_location='cpu')
    net=learned_net(len(ckpt['mu']));net.load_state_dict(ckpt['weights']);net.eval()
    with torch.inference_mode():prob=torch.sigmoid(net(((v-ckpt['mu'])/ckpt['sd']).clamp(-6,6))[1]).numpy()
    v,y=v.numpy(),y.numpy()
    known=(v[:,-8]==0)&(v[:,-7]==0)
    # Use group-macro top-1 on paths entirely supported by observed free cells.
    records=[]
    for alpha in (.5,1.,2.):
        for ratio in (0.,.25):
            selected=[]
            for i in range(0,len(v),28):
                ind=np.flatnonzero(known[i:i+28])+i
                if not len(ind):continue
                utility=prob[ind,1]*(prob[ind,0]**alpha)-ratio*v[ind,-2]
                selected.append(int(ind[np.argmax(utility)]))
            records.append(dict(alpha=alpha,cost_penalty=ratio,groups=len(selected),
                       reached=int(y[selected,0].sum()),clear=int(y[selected,1].sum())))
    best=max(records,key=lambda r:(r['clear'],r['reached'],-r['cost_penalty'],-abs(r['alpha']-1)))
    report=dict(rule='Visible and reachable endpoints first, then physical traversal, validated on sampled routes with observed centreline clearance throughout. Ties use simpler scores.',chosen=best,
                calibration_grid=records,known_free_candidates=int(known.sum()),total=len(known),
                reached_fraction_known=float(y[known,0].mean()) if known.any() else None,
                reached_fraction_all=float(y[:,0].mean()))
    assert best['groups']>0
    (HERE/'calibration.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
