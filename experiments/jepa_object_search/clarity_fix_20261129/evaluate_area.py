"""Blind first-accept replay, followed by a separate mask-based scoring pass."""
from collections import Counter,defaultdict
import json,os,sys
import numpy as np
from area_verifier import AreaMemory,HERE,REFINED
from score_paired import box_grade

STRICT='--strict' in sys.argv
DECISIONS=HERE/'private'/('strict_decisions' if STRICT else 'decisions')

def replay(path):
    with np.load(path) as s:
        z,rgb,pose,source,budget=[s[k] for k in ('depth','rgb','odom','cue','budget')]
    if STRICT:
        from strict_rescue import StrictRescueMemory
        factory=StrictRescueMemory
    else:
        factory=AreaMemory
    memory=factory(rgb[0],z[0],source)
    assert len(rgb)-1<=int(budget)
    t=None;box=None
    for i,(r,d,p) in enumerate(zip(rgb,z,pose)):
        e=memory.observe(r,d,p)
        if e['accept']:
            assert e['p_clear']*e['p_same']>=.835
            t=i;box=list(map(int,e['box']));break
    dest=DECISIONS/path.parent.name/(path.stem+'.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(dict(step=t,box=box,actions_used=i)))

def grade(rows,mask):
    if rows['step'] is None:return 'unreported'
    n,g,j=box_grade(mask[rows['step']],rows['box'])
    return 'correct' if n>=369 and g>=.5*n and j>=.30 else 'false'

def score():
    records=sorted(DECISIONS.glob('*/*.json'))
    assert len(records)==94
    table=defaultdict(Counter);train29=defaultdict(Counter);new_early=Counter();per_house=defaultdict(Counter)
    focused14=Counter()
    totals=Counter();old=Counter();cost=0;new_fp=Counter()
    for rec in records:
        house=rec.parent.name; cue=rec.stem
        data=json.loads(rec.read_text())
        with np.load(REFINED/'private_scoring'/house/(cue+'.npz')) as g:
            masks=np.unpackbits(g['packed'],axis=1).reshape(-1,192,192)
        olddata=json.loads((REFINED/'private_paired'/'full'/house/cue/'decision.json').read_text())
        origin=json.loads((REFINED/'private_paired'/'old'/house/cue/'decision.json').read_text())
        a,b=grade(olddata,masks),grade(data,masks)
        c=grade(origin,masks)
        table[a][b]+=1;totals[b]+=1;old[a]+=1;cost+=data['actions_used'];per_house[house][b]+=1
        if c=='unreported' and masks.reshape(len(masks),-1).sum(1).max()>=369:
            train29[a][b]+=1
            if a=='unreported':
                eligible=[]
                for e,mask in zip(olddata['trace'],masks):
                    if e['box'] is None:continue
                    n,i,j=box_grade(mask,e['box'])
                    if n>=369 and i>=n*.5 and j>=.3:eligible.append(e)
                if eligible:
                    best=max(eligible,key=lambda v:v['p_same']*v['p_clear'])
                    if best['p_clear']<.835<=best['p_same']:focused14[b]+=1
        if a=='false' and c!='false' and 0<masks[olddata['step']].sum()<369:new_early[b]+=1
        if b=='false':
            n,i,j=box_grade(masks[data['step']],data['box'])
            new_fp['below_area' if n<369 else 'box_mismatch']+=1
    assert sum(map(lambda c:sum(c.values()),train29.values()))==29
    assert sum(focused14.values())==14 and sum(new_early.values())==5
    report=dict(paired_requests=94,previous=old,area_head=totals,paired_transitions=table,old_29_clear_unreported_transitions=train29,
                newest_subthreshold_previously_not_false=new_early,previous_score_only_14=focused14,incorrect_reason=new_fp,
                actions=cost,per_house=[dict(v) for k,v in sorted(per_house.items())],
                protocol='Previous 94 fixed routes, 22/26/30 physical actions. First accepted frame; outcome scored separately using native target mask and original rules.')
    report['variant']='strict_two_view_auxiliary' if STRICT else 'direct_area_replacement'
    (HERE/('strict_paired.json' if STRICT else 'area_paired.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    os.umask(0o077)
    if sys.argv[1]=='score':score()
    else:
        from multiprocessing import Pool
        paths=list(sorted((REFINED/'private_inputs').glob('*/*.npz')))
        with Pool(int(sys.argv[1])) as pool:pool.map(replay,paths)
