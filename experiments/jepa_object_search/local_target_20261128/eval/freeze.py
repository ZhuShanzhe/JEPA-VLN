"""Freeze episodes by category order, salted hash, two connected instances, and unused route-development house."""
import gzip
import json
import math
import hashlib
import shutil
import sys
from pathlib import Path
from inventory import HERE,P,BASE,DATA,sha,read,inventory

GOALS='clock|wall clock|painting|picture|cup|mug|table lamp|bed|chair|table|plant|sofa|shelf|television|stool|office chair|armchair|cabinet|pillow'.split('|')


def rank(row):
    return hashlib.sha256(('local-target-independent-v1:'+row['scene']+':'+str(row['episode_id'])).encode()).digest()


def write_once(path,data):
    if path.exists():
        assert read(path)==data, path
    else:
        with path.open('x') as out:json.dump(data,out,indent=2)


def main():
    old=BASE/'eval4'
    lock=read(old/'locked_sources.json')
    for rel,digest in lock.items():assert sha(P/rel)==digest,rel
    for f in ('physical_panel.py','inference_worker.py'):
        if not (HERE/f).exists():shutil.copyfile(old/f,HERE/f)
        assert sha(HERE/f)==sha(old/f)
    if not (HERE/'locked').exists():
        shutil.copytree(old/'locked',HERE/'locked',ignore=shutil.ignore_patterns('__pycache__'))
    for f in (HERE/'locked').rglob('*'):
        if f.is_file() and f.suffix!='.pyc': assert sha(f)==sha(old/f.relative_to(HERE))
    from physical_panel import sensor_sim
    sys.path.insert(0,str(P/'src/stopwatch'))
    from stopwatch.eval.geometry import _load_goal_sets,load_scene_instances,geodesic_distance
    import numpy as np
    houses,previous,used=inventory()
    unrelated=P.parent/'jepa_vln_paper_bench_20261017/results/fgprompt_place_compact430/episodes'
    unrelated_used=sorted(s for s in houses if any(unrelated.glob('hssd-place-'+s+'-*')))
    assert houses-used,'No eligible houses'
    write_once(HERE/'frozen_exposure.json',dict(available=sorted(houses),prior_related=previous,eligible_pool=sorted(houses-used),
               different_project_october_place_baseline=unrelated_used))
    goals=_load_goal_sets(DATA)['scoring_class_ids']['hssdov']
    train_targets=read(HERE/'locked/learned_prior.json')['targets']
    rows=[json.loads(l) for l in (DATA/'episode_goal_join.jsonl').read_text().splitlines()]
    panel=[]; audit=[]
    assert houses-used,'No eligible houses'
    for goal in GOALS:
        if goal.replace(' ','_') not in train_targets:continue
        for row in sorted((r for r in rows if r['category']==goal and r['scene'] not in used and r['goal_instances']>=2),key=rank):
            shard=DATA/'eval/hssdov/content'/(row['scene']+'.json.gz')
            ep=next(e for e in json.load(gzip.open(shard))['episodes'] if str(e['episode_id'])==str(row['episode_id']))
            sim=sensor_sim(ep)
            try:
                instances=load_scene_instances(DATA/'protocol/instance_viewpoints/hssd','val',ep['scene_id'])
                distances=[]
                for r in instances:
                    if r['class_id'] in goals[goal] and r['view_points']:
                        pts=np.float32([v['agent_state']['position'] for v in r['view_points']])
                        distances.append(geodesic_distance(sim,ep['start_position'],pts))
            finally:sim.close()
            conn=sum(map(math.isfinite,distances))
            # Match eval4's connectedness gate; starting proximity is audited and reported per arm.
            audit.append(dict(scene=row['scene'],id=ep['episode_id'],goal=goal,reachable=conn,initially_close=sum(d<.25 for d in distances)))
            if conn>=2:
                panel.append(dict(scene=row['scene'],id=ep['episode_id'],goal=goal,quota=2,budget=300,shard_sha256=sha(shard)))
                used.add(row['scene'])
                break
        if len(panel)==8:break
    assert len(panel)>=1
    assert len({r['scene'] for r in panel})==len(panel)
    write_once(HERE/'eligibility.json',audit)
    write_once(HERE/'panel.json',panel)
    record={str(f.relative_to(P)):sha(f) for f in [HERE/'panel.json',HERE/'frozen_exposure.json',HERE/'eligibility.json',HERE/'freeze.py',
        HERE/'inventory.py',old/'locked_sources.json',HERE/'physical_panel.py',HERE/'inference_worker.py',*(f for f in (HERE/'locked').rglob('*') if f.is_file() and f.suffix!='.pyc')]}
    write_once(HERE/'panel_lock.json',record)
    print('FROZEN',len(panel),sha(HERE/'panel.json'),flush=True)

if __name__=='__main__':main()
