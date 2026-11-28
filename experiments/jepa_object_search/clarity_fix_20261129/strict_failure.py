"""After scoring, count why the auxiliary gate could not confirm the fourteen known clarity misses."""
from collections import Counter
import json
import numpy as np
from area_verifier import HERE,REFINED,RefinedMemory
from strict_rescue import measure,strict_settings
from score_paired import box_grade

def main():
    counts=Counter();distances=[]
    cfg=strict_settings()
    for f in sorted((REFINED/'private_inputs').glob('*/*.npz')):
        h=f.parent.name;c=f.stem
        old=json.loads((REFINED/'private_paired'/'full'/h/c/'decision.json').read_text())
        if old['step'] is not None:continue
        original=json.loads((REFINED/'private_paired'/'old'/h/c/'decision.json').read_text())
        if original['step'] is not None:continue
        with np.load(REFINED/'private_scoring'/h/f.name) as d:
            masks=np.unpackbits(d['packed'],axis=1).reshape(-1,192,192)
        eligible=[]
        for mask,e in zip(masks,old['trace']):
            if e['box'] is None:continue
            n,i,j=box_grade(mask,e['box'])
            if n>=369 and i>=n*.5 and j>=.3:eligible.append(e)
        if not eligible:continue
        best=max(eligible,key=lambda e:e['p_same']*e['p_clear'])
        if not best['p_clear']<.835<=best['p_same']:continue
        with np.load(f) as d:
            rgb,z,odo,box=[d[k] for k in ('rgb','depth','odom','cue')]
        mem=RefinedMemory(rgb[0],z[0],box);valid=[];scores=0
        for t,(im,dep,p) in enumerate(zip(rgb,z,odo)):
            _,ev=measure(mem,im,dep,p)
            if ev and ev[2]:
                scores+=1
                if ev[1]>=cfg['min_area']:
                    valid.append((t,p[:2]))
        spread=max([float(np.linalg.norm(p-q)) for i,p in valid for j,q in valid if i<j<=i+8] or [0.])
        if scores==0:counts['no_current_view_crossed_joint_score']+=1
        elif not valid:counts['no_view_also_supported_by_both_area_estimates']+=1
        elif len(valid)==1:counts['only_one_qualifying_view']+=1
        elif spread<cfg['min_displacement']:counts['qualifying_views_missing_required_translation_within_8_frames']+=1
        else:counts['can_confirm']+=1
        if valid:distances.append(spread)
    assert sum(counts.values())==14
    doc=dict(diagnostic_groups=14,first_blocker=counts,max_available_baseline_m_for_groups_with_a_qualifying_view=distances)
    (HERE/'strict_failure.json').write_text(json.dumps(doc,indent=2)+'\n')
    print(doc)
if __name__=='__main__':main()
