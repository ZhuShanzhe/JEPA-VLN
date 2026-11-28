"""Two-view auxiliary confirmation; the primary decision, memory and identity are retained."""
import json
from functools import lru_cache
import numpy as np
from area_verifier import (AreaEstimate, HERE, RefinedMemory, crop_features, pixel_features,
                           frame_features, combine)
from frame_confirm import thresholds
from memory_refit import relative

@lru_cache(None)
def strict_settings():
    return json.loads((HERE/'private'/'strict_gate.json').read_text())

def measure(memory,rgb,depth,pose):
    out=memory.observe(rgb,depth,pose)
    base=bool(out['accept'])
    # Reanchored candidates may originate in either source; never corroborate
    # a pair against different anchors. These retain original handling.
    if base or memory.state!='trusted_initial' or out['box'] is None:
        return out, None
    ref=memory.verifier
    rel=relative(pose,memory.anchor)
    old=ref.observe(rgb,depth,rel);box=old['box']
    if box is None or tuple(out['box'])!=tuple(box):return out,None
    m=AreaEstimate.load()
    assert m.cfg['area_weight']==m.cfg['old_weight']==0
    px,_,_,inside,scale=pixel_features(rgb,depth,box,ref,rel)
    px=px.astype('float16').astype('float32')
    fg=m.pixel(px)
    orig=crop_features(rgb,depth,box,ref,rel,old)
    x=frame_features(fg,px,inside,scale,orig)[None]
    p=float(m.area(x)[0])
    reg=370*p/max(1e-8,1-p)-1
    pc=float(combine(m.binary(x)[0],out['p_clear'],0.,m.cfg['fusion']))
    valid=pc*out['p_same']>=thresholds()['threshold']
    return out,(pc, float(min(reg,fg.sum()*scale)),valid)

class Agreement:
    def __init__(self):
        self.history=[]
        self.index=0
    def accept(self,pose,record,cfg):
        i=self.index
        self.index+=1
        self.history=[(j,q) for j,q in self.history if i-j<=8]
        if record is None or not record[2] or record[1]<cfg['min_area']:return False
        xz=np.asarray(pose[:2],float)
        matched=any(np.linalg.norm(xz-q)>=cfg['min_displacement'] for j,q in self.history)
        self.history.append((i,xz.copy()))
        return matched

class StrictRescueMemory:
    def __init__(self,rgb,depth,box):
        self.primary=RefinedMemory(rgb,depth,box)
        self.agreement=Agreement()
        self.cfg=strict_settings()
    def observe(self,rgb,depth,pose):
        out,ev=measure(self.primary,rgb,depth,pose)
        if self.agreement.accept(pose,ev,self.cfg):
            out=dict(out,p_clear=ev[0],accept=True,reason='same_anchor_two_displaced_clear_views')
        return out
