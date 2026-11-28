"""Calibrate the initial-memory gate on six validation buildings with a first-confirmation retention constraint."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import hashlib,json,os
import numpy as np
from train_memory import sample_files,read,results
from memory_refit import HERE,MODELS,Surface,FrameConfirmer,model,thresholds

from train_memory import variants,fit,seed
from memory_refit import RefinedMemory,risk_features

def danger(m,box,tracker):
    l,t,r,b=box;sy,sx=np.mgrid[t:b,l:r]
    stride=max(1,int(np.ceil(sx.size/6000)))
    fg=m[sy.ravel()[::stride],sx.ravel()[::stride]][tracker.w>=.35]
    purity=fg.mean() if len(fg) else 0.
    inside=m[t:b,l:r].sum()
    return bool(purity<.43 or (inside<125 and inside/((r-l)*(b-t))<.27)),float(purity)

def risk_train(path):
    rgb,dep,_,m,box=read(path);rows=[]
    for b in variants(box,seed(path)%2):
        s=Surface(rgb[0],dep[0],b);v=FrameConfirmer(rgb[0],dep[0],b)
        bad,_=danger(m[0],b,v)
        rows.append((risk_features(s,v),bad))
    return rows

def assess(path):
    rgb,dep,poses,masks,box=read(path)
    # The gate-independent two possible decisions are i) keep original and ii) delay.
    # Both consume the physical continuous frames in order and stop on their first acceptance.
    native=RefinedMemory(rgb[0],dep[0],box,use_memory=False,use_clarity=False)
    pending=RefinedMemory(rgb[0],dep[0],box,use_memory=False,use_clarity=False)
    pending.source=Surface(rgb[0],dep[0],box)
    pending.initial_confidence=float(model('initial')(np.asarray([pending.source.summary()]))[0])
    pending.contamination_risk=float(model('contamination')(np.asarray([risk_features(pending.source,pending.verifier)]))[0])
    pending.state='pending';s=pending.source
    dangerous,_=danger(masks[0],box,native.verifier)
    baseline=None; promoted=None; any_translation=False; independently_reliable=False
    for i,p in enumerate(poses):
        any_translation |= bool(np.hypot(*p[:2])>=.12)
        if baseline is None:
            d=native.observe(rgb[i],dep[i],p)
            if d['accept']:baseline=all(results(masks[i],d['box']))
        if promoted is None:
            d=pending.observe(rgb[i],dep[i],p)
            independently_reliable |= pending.state=='reanchored'
            if d['accept']:promoted=all(results(masks[i],d['box']))
    return pending.initial_confidence,pending.contamination_risk,dangerous,baseline,promoted,any_translation,independently_reliable

def atomic_json(destination,value):
    payload=json.dumps(value,indent=2,default=lambda item:item.item())+'\n'
    temp=destination.with_suffix('.partial')
    temp.write_text(payload);temp.replace(destination)

def run():
    os.umask(0o077);info=json.loads((HERE/'memory_fit.json').read_text())
    gates=json.loads((MODELS/'gates.json').read_text())
    original=info['calibration'].get('native_retention',{}).get('original_purity_gate',gates['initial'])
    train=sample_files('train')
    with ProcessPoolExecutor(max_workers=6) as pool: samples=[r for z in pool.map(risk_train,train) for r in z]
    x,y=map(np.asarray,zip(*samples));fit('contamination',x,y)
    paths=sorted((HERE/'private_memory_validation').glob('*/*.npz'))
    assert len(paths)==144 and len({x.parent.name for x in paths})==6
    with ProcessPoolExecutor(max_workers=6) as pool: rows=list(pool.map(assess,paths))
    before=sum(b is True for _,_,_,b,*_ in rows)
    def stats(t,h):
        st=Counter();st['natural']=len(rows)
        for trust,risk,bad,old,new,moved,relook in rows:
            hold=trust<t and risk>=h
            st['severe_initial_contamination']+=bad
            if hold:
                st['held']+=1;st['unsafe_held']+=bad;st['could_translate']+=moved
                st['promoted']+=relook;st['promoted_submitted']+=new is not None;st['promoted_correct']+=new is True
            outcome=new if hold else old
            st['previous_correct']+=old is True;st['preserved_correct']+=old is True and outcome is True
            st['previous_correct_held']+=old is True and hold
            st['previous_correct_held_recovered']+=old is True and hold and new is True
            st['previous_incorrect']+=old is False;st['new_incorrect']+=outcome is False
        return st
    opts=[]
    for t in (.15,.3,.5,.7,float(original),1.01):
        for h in np.arange(.1,.99,.02):
            s=stats(t,h)
            if s['unsafe_held']<3 or s['held']<4:continue
            retained=s['preserved_correct']/max(1,before)
            eligible=retained>=.95 and s['new_incorrect']<=s['previous_incorrect']
            cost=4*(s['previous_correct']-s['preserved_correct'])+3*max(0,s['new_incorrect']-s['previous_incorrect'])
            rank=(eligible,s['unsafe_held']-cost-0.5*(s['held']-s['unsafe_held']),s['preserved_correct'],float(h),-t)
            opts.append((rank,t,float(h),s))
    _,best_t,best_h,final=max(opts,key=lambda r:r[0])
    gates.update(initial=best_t,contamination=best_h)
    atomic_json(MODELS/'gates.json',gates)
    info['thresholds'].update(initial=best_t,contamination=best_h)
    info['train']['severe_contamination_training_labels']=int(y.sum())
    info['calibration']['native_retention']={'original_purity_gate':original,'purity_gate_only_counts':stats(original,0.),'final_counts':dict(final),'native_total':len(paths),
        'protocol':'144 natural validation cues in six houses. Every path comprises a single physical run of 22 steps, with a fixed turn-and-retry after contact. In the delayed arm the first independently qualified translated view remains the sole new reference for following observations. Both arms use the same prior clarity function and stop on first formal acceptance. Select a non-degenerate contamination gate, favor retaining 95% of old correct confirmations without extra false submissions.'}
    info['hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(MODELS.iterdir())}
    info['calibration']['initial_metrics_scope']='Initial purity-only calibration statistics apply to original_purity_gate. Final two-score admission and sequential verification on natural cues are in native_retention.'
    atomic_json(HERE/'memory_fit.json',info)
    print(info['calibration']['native_retention'],flush=True)
if __name__=='__main__':run()
