"""Lock an auxiliary two-view gate on six calibration houses, before the independent panel."""
import json,sys,os
from multiprocessing import Pool
from collections import Counter
from hashlib import sha256
import numpy as np
from area_verifier import HERE,REFINED,RefinedMemory
from strict_rescue import Agreement,measure
from prepare_clarity import all_files
from score_paired import box_grade

def journal(path):
    with np.load(path) as d:
        rgb,z,pose,box,flags=[d[k] for k in ('rgb','depth','pose','box','mask')]
    model=RefinedMemory(rgb[0],z[0],box)
    rows=[]
    for im,depth,p in zip(rgb,z,pose):
        out,extra=measure(model,im,depth,p)
        rows.append((p[:2].astype(float).tolist(),bool(out['accept']),extra,out['box']))
        if out['accept']:break
    result=[]
    for t,(pos,accepted,extra,rect) in enumerate(rows):
        good=False
        if rect is not None:
            mask=np.unpackbits(flags[t]).reshape(192,192)
            area,cover,iou=box_grade(mask,rect)
            good=bool(area>=369 and cover>=area/2 and iou>=.3)
        result.append((pos,accepted,extra,good))
    return result

def outcome(seq,cfg):
    state=Agreement()
    for pos,base,extra,good in seq:
        if base or cfg is not None and state.accept(pos,extra,cfg):
            return 'correct' if good else 'false'
    return 'unreported'

if __name__=='__main__':
    os.umask(0o077)
    paths=[a for a,b in all_files() if b.parent.parent.name=='val']
    assert len(paths)==288
    with Pool(6) as pool: seq=pool.map(journal,paths)
    old=[outcome(s,None) for s in seq]
    baseline=Counter(old)
    opts=[]
    pool=[]
    for dx in (.20,.40,.60):
        for area in (369.,425.,480.,550.):
            cfg=dict(min_area=area,min_displacement=dx)
            new=[outcome(s,cfg) for s in seq]
            changed=Counter(a+' to '+b for a,b in zip(old,new) if a!=b)
            lost=sum(a=='correct' and b!='correct' for a,b in zip(old,new))
            errors=sum(a!='false' and b=='false' for a,b in zip(old,new))
            wins=sum(a!='correct' and b=='correct' for a,b in zip(old,new))
            # A zero-regression side-channel is preferable even with fewer gains.
            key=(not (lost or errors),-lost-errors,wins,area,dx)
            pool.append((key,cfg,new));opts.append(dict(config=cfg,transitions=changed))
    _,cfg,selected=max(pool,key=lambda t:t[0])
    freeze=HERE/'private'/'strict_gate.json'
    freeze.write_text(json.dumps(cfg)+'\n')
    report=dict(calibration_houses=6,calibration_sweeps=len(seq),comparison='Full original memory with original camera-area head, earliest valid decision on each physically acquired motion sweep.',
                old=baseline,selected=Counter(selected),gate=cfg,calibration_grid=opts,
                model_sha256=json.loads((HERE/'area_training.json').read_text())['sha256'],
                gate_sha256=sha256(freeze.read_bytes()).hexdigest())
    (HERE/'strict_calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(baseline,cfg,report['selected'],flush=True)
