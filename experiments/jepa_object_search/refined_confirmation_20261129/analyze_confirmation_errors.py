"""Diagnose first rejected or wrong target evidence after inference, without sweep-specific thresholds."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from score_paired import HERE,box_grade


def grade(mask,box):
    if box is None:return False
    size,pix,iou=box_grade(mask,box)
    return size>=369 and pix>=.5*size and iou>=.30


def audit(arm):
    ct=Counter();old_losses=Counter();incorrect=Counter();rem=Counter();limited=Counter()
    changes=Counter();new_false_pixels=[];left_sizes=[];source_flags=Counter()
    for p in sorted((HERE/'private_paired'/arm).glob('*/*/decision.json')):
        house,src=p.parent.parent.name,p.parent.name
        old_record=json.loads((HERE.parent/'reactive_view_20261129/private_roll'/house/(src+'.json')).read_text())
        dec=json.loads(p.read_text());t=dec['step'];tr=dec['trace']
        with np.load(HERE/'private_scoring'/house/(src+'.npz')) as f:
            masks=np.unpackbits(f['packed'],axis=1).reshape(-1,192,192)
        ot=old_record['new_confirmation'];old=old_record['events']
        history=('correct' if ot is not None and old[ot]['correct'] else 'false' if ot is not None else
                 'miss_seen' if any(v['px']>=369 for v in old) else 'unseen')
        outcome='absent' if t is None else 'correct' if grade(masks[t],tr[-1]['box']) else 'false'
        ct[history+' to '+outcome]+=1
        if history=='correct' and outcome!='correct':
            reason=('earlier_false_stop' if t is not None and t<ot else
                    'still_good_box_at_original_confirm_step' if grade(masks[ot],tr[ot]['box']) else
                    'box_missing_at_original_confirm_step' if tr[ot]['box'] is None else 'original_box_now_invalid')
            old_losses[reason]+=1
        if outcome=='false':
            size,pix,iou=box_grade(masks[t],tr[-1]['box'])
            reason='empty_target_box' if pix==0 else 'size_under_369' if size<369 else 'poor_box'
            incorrect[reason]+=1
            if history!='false':
                changes[history+' to '+reason]+=1
                new_false_pixels.append(size)
                source_flags[tr[0]['state']]+=1
        if history=='miss_seen' and outcome=='absent':
            valid=[e for e,mask in zip(tr,masks) if grade(mask,e['box'])]
            if not valid:
                rem['no_valid_candidate_at_clear_frame']+=1
                continue
            best=max(valid,key=lambda e:e['p_same']*e['p_clear'])
            left_sizes.append(int(masks[best['step']].sum()))
            a=best['p_clear']<.835;b=best['p_same']<.835
            rem['valid_box_both_low' if a and b else 'valid_box_clarity_low' if a else
                'valid_box_identity_low' if b else 'valid_box_product_low']+=1
        if history=='miss_seen' and outcome=='correct':
            limited['at_source_or_scan']+=t<=2
            limited['after_old_actual_endpoint']+=t>old_record['baseline_length']
    assert sum(ct.values())==94
    return dict(transitions=ct,reason_previously_correct_now_lost=old_losses,
                first_incorrect=incorrect,new_false_by_group_and_cause=changes,
                new_false_ground_truth_pixels_summary={} if not new_false_pixels else dict(
                    minimum=int(min(new_false_pixels)),median=float(np.median(new_false_pixels)),maximum=int(max(new_false_pixels))),
                new_false_initial_reference_state=source_flags,
                residual_of_old_29_with_no_submission=rem,
                residual_best_usable_frame_true_area_median=float(np.median(left_sizes)) if left_sizes else None,
                recovered_position=limited)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('arm', choices=['clarity','memory','full']);args=ap.parse_args()
    result=audit(args.arm)
    (HERE/(args.arm+'_error_diagnostics.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
