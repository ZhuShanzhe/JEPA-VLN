"""Estimate the visible size of the remembered instance from the current camera crop."""
import json
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent/'reactive_view_20261129'
sys.path.insert(0,str(PRIOR))
from frame_confirm import Trees, load as legacy, source_features


def crop_features(rgb, depth, box, reference, pose, old):
    x,y,r,b=map(int,box)
    depth=np.asarray(depth).squeeze().astype('float32')
    rgb=np.asarray(rgb)[...,:3]
    # Integrate foreground probabilities on the *current* crop as an independent area cue.
    z=depth[y:b,x:r]
    raw,cols,rows=source_features(rgb,depth,box)
    ix=np.linspace(0,len(raw)-1,min(196,len(raw))).astype(int)
    local=legacy('source')(raw[ix])
    local[depth[rows[ix],cols[ix]]<=.05]=0
    weights=reference.w
    net_area=max(1.,float(reference.start_area))
    srcbox=reference.sourcebox
    initial_size=max(1,(srcbox[2]-srcbox[0])*(srcbox[3]-srcbox[1]))
    crop_area=(r-x)*(b-y)
    rgb_crop=rgb[y:b,x:r].astype('float32')
    h,w=z.shape
    core=rgb_crop[h//4:max(h//4+1,(3*h+3)//4),w//4:max(w//4+1,(3*w+3)//4)]
    ring=depth[max(0,y-3):min(192,b+3),max(0,x-3):min(192,r+3)]
    ref_depth=reference.xyz[:,2]
    keep=weights>.35
    ref_med=float(np.median(ref_depth[keep])) if keep.any() else 0.
    def q(values, levels):
        data=np.asarray(values)
        return np.quantile(data,levels) if data.size else np.zeros(len(levels))
    qa=q(z[z>.05],[.1,.25,.5,.75,.9])
    local_area=crop_area*local.mean()
    estimated=old['estimated_pixels']
    feats=[estimated,crop_area,net_area,initial_size,net_area/initial_size,
           r-x,b-y,(r-x)/max(1,b-y),local_area,local.mean(),local.std(),(local>.7).mean(),(local>.9).mean(),
           estimated/max(1,crop_area),estimated/max(1,local_area),local_area/max(1,net_area),
           net_area*min(7.,ref_med/max(.05,qa[2]))**2,
           float(x<2 or y<2 or r>190 or b>190),abs(pose[0]),pose[1],abs(pose[2]),ref_med,
           *qa,qa[3]-qa[1],qa[-1]-qa[0],
           *q(ring[ring>.05],[.1,.5,.9]), *np.std(rgb_crop,axis=(0,1)),
           *np.mean(abs(rgb_crop-np.median(core.reshape(-1,3),axis=0)),axis=(0,1)),
           *(np.mean(abs(np.diff(rgb_crop,axis=1)),axis=(0,1)) if w>1 else (0.,0.,0.))]
    return np.nan_to_num(np.asarray(feats,dtype=np.float32))


def blend(old_p, image_p, area_p, weight, floor):
    # A calibrated fusion of independent current-view area and foreground estimates.
    score=(1-weight)*image_p+weight*area_p
    return np.maximum(score,floor*old_p)


class ClarityEstimator:
    @classmethod
    @lru_cache(1)
    def load(cls):
        self=cls()
        directory=HERE/'private_clarity_model'
        self.pixel=Trees(directory/'pixel_area.npz')
        self.clear=Trees(directory/'clear_frame.npz')
        self.cal=json.loads((directory/'calibration.json').read_text())
        return self

    def predict(self,rgb,depth,candidate_box,ref_verifier,pose,old_evidence):
        if candidate_box is None: return 0.
        x=crop_features(rgb,depth,candidate_box,ref_verifier,pose,old_evidence)[None]
        estimated=np.clip(self.pixel(x),1e-8,1-1e-8)
        area_logit=np.log(estimated/(1-estimated))
        area_p=1/(1+np.exp(-np.clip(self.cal['slope']*area_logit+self.cal['intercept'],-40,40)))
        return float(blend(old_evidence['p_clear'],self.clear(x)[0],area_p[0],self.cal['area_weight'],self.cal['old_floor']))
