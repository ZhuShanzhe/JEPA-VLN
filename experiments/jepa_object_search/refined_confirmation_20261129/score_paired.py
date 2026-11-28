"""Grade only the first saved evidence for each request. Simulator labels are read here, after decisions."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
MODELS=('old','clarity','memory','full')


def box_grade(mask, box):
    l,t,r,b=box
    pixels=int(mask.sum())
    overlap=int(mask[t:b,l:r].sum())
    yy,xx=np.where(mask)
    if len(yy):
        xl,xr=int(xx.min()),int(xx.max())+1
        yt,yb=int(yy.min()),int(yy.max())+1
        intersection=max(0,min(xr,r)-max(xl,l))*max(0,min(yb,b)-max(yt,t))
        iou=intersection/((xr-xl)*(yb-yt)+(r-l)*(b-t)-intersection)
    else:
        iou=0.
    return pixels,overlap,iou


def audit_one(labels, cue_record, sensors, directory, arm):
    document=json.loads((directory/'decision.json').read_text())
    step=document['step']
    trace=document['trace']
    assert document['actions_used']==len(trace)-1<=cue_record['budget']
    assert not any(x['accept'] for x in trace[:-1])
    old_step=cue_record['new_confirmation']
    if arm=='old':
        assert step==old_step
        if step is not None:assert trace[-1]['box']==cue_record['events'][step]['box']
    correct=False
    bad=None
    if step is None:
        assert not any(x['accept'] for x in trace)
        assert document['actions_used']==len(labels)-1
    else:
        assert step==len(trace)-1 and trace[-1]['accept']
        box=tuple(document['box'])
        l,t,r,b=box
        with Image.open(directory/'frame.png') as f, Image.open(directory/'crop.png') as c:
            image=np.asarray(f)
            assert np.array_equal(np.asarray(c),image[t:b,l:r]) and np.array_equal(image,sensors[step])
            assert hashlib.sha256(image.tobytes()).hexdigest()==document['frame_sha256']
        pixels,inter,iou=box_grade(labels[step],box)
        correct=bool(pixels>=369 and inter>=.5*pixels and iou>=.30)
        if not correct:
            bad=('target_absent_from_box' if inter==0 else
                 'target_too_small' if pixels<369 else 'wrong_localization')
    if arm=='old':
        assert correct==(old_step is not None and cue_record['events'][old_step]['correct'])
    return ('correct' if correct else 'false' if step is not None else 'unreported',
            document['actions_used'], bad)


def main():
    houses=json.loads((HERE.parent/'new_view_20261129'/'view_design.json').read_text())['cohorts']['blind']
    totals=defaultdict(Counter)
    transitions={k:defaultdict(Counter) for k in MODELS}
    by_house=[]
    wrong=defaultdict(Counter)
    by_memory=defaultdict(Counter)
    rescue_overlap=Counter()
    transitions_versus_clarity=defaultdict(Counter)
    missed_29=Counter()
    for house in houses:
        local={arm:Counter() for arm in MODELS}
        for f in sorted((HERE/'private_scoring'/house).glob('*.npz')):
            h=json.loads((HERE.parent/'reactive_view_20261129'/'private_roll'/house/(f.stem+'.json')).read_text())
            assert h['focus']
            with np.load(f) as gt:
                labels=np.unpackbits(gt['packed'],axis=1).reshape(-1,192,192)
            with np.load(HERE/'private_inputs'/house/f.name) as inputs:
                frames=inputs['rgb']
                assert len(labels)==len(frames)==len(h['events'])
                assert len(frames)-1<=int(inputs['budget'])==h['budget']
            t=h['new_confirmation']
            history=('old_correct' if t is not None and h['events'][t]['correct'] else
                     'old_false' if t is not None else 'old_clear_unreported' if any(v['px']>=369 for v in h['events']) else
                     'old_never_clear_unreported')
            outcomes={}
            for arm in MODELS:
                outcome,actions,error=audit_one(labels,h,frames,HERE/'private_paired'/arm/house/f.stem,arm)
                outcomes[arm]=outcome
                transitions[arm][history][outcome]+=1
                totals[arm][outcome]+=1
                totals[arm]['actions']+=actions
                local[arm][outcome]+=1
                if error:wrong[arm][error]+=1
            result_path=HERE/'private_paired'/'full'/house/f.stem/'decision.json'
            full_trace=json.loads(result_path.read_text())['trace']
            first_state,last_state=full_trace[0]['state'],full_trace[-1]['state']
            by_memory[first_state+' to '+last_state][history+' to '+outcomes['full']]+=1
            transitions_versus_clarity[outcomes['clarity']][outcomes['full']]+=1
            if history=='old_clear_unreported':
                combo=' + '.join(k for k in ('clarity','memory','full') if outcomes[k]=='correct')
                rescue_overlap[combo or 'none']+=1
                if outcomes['full']=='unreported':
                    seen=[(entry,mask) for entry,mask in zip(full_trace,labels) if mask.sum()>=369]
                    assert seen
                    admitted=[(entry,mask) for entry,mask in seen if entry['state']!='pending']
                    eligible=[]
                    for entry,mask in admitted:
                        if entry['box'] is None:
                            continue
                        size,inside,iou=box_grade(mask,entry['box'])
                        if inside>=.5*size and iou>=.30:
                            eligible.append(entry)
                    if not admitted: cause='memory_pending_on_all_clear_frames'
                    elif not eligible: cause='no_usable_match_on_admitted_clear_frames'
                    else:
                        best=max(eligible,key=lambda z:z['p_clear']*z['p_same'])
                        low_s=best['p_same']<.835
                        low_c=best['p_clear']<.835
                        cause=('low_identity_and_clarity' if low_s and low_c else
                               'low_clarity' if low_c else 'low_identity' if low_s else 'product_below_cutoff')
                    missed_29[cause]+=1
        by_house.append(local)
    old=totals['old']
    assert (old['correct'],old['false'],old['unreported'],old['actions'])==(26,6,62,1855)
    assert {k:sum(v.values()) for k,v in transitions['old'].items()}=={
        'old_correct':26,'old_false':6,'old_clear_unreported':29,'old_never_clear_unreported':33}
    for arm in MODELS:
        assert sum(totals[arm][k] for k in ('correct','false','unreported'))==94
    result=dict(groups=94, results=totals, transition_from_previous=transitions,
                wrong_submissions=wrong, per_house=by_house, full_memory_state_then_outcome=by_memory,
                full_outcome_by_clarity_only_outcome=transitions_versus_clarity,
                old_clear_unreported_success_overlap=rescue_overlap,
                remaining_abstentions_among_29=missed_29,
                protocol='Fixed 94 paired requests, 22/26/30-action caps, identical physically replayed motion up to first acceptance. Native RGB-D and relative odometry for decisions. Target masks opened after lock for scoring only. First submission must see at least 369 same-instance pixels, cover >=50% and reach box IoU >=0.30.')
    (HERE/'paired_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
