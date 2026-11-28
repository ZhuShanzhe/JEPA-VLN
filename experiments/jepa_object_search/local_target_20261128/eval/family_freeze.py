"""Freeze a building-family-disjoint panel without inspecting outcomes."""
import gzip
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from inventory import HERE,P,BASE,DATA,sha,read
from freeze import GOALS,rank,write_once

OUT=HERE.parent/'eval_family'


def parsed_ids(text):
    return set(re.findall(r'(?<!\d)10\d{7}(?:_\d{9})?(?!\d)',text))


def family(scene):
    return scene.split('_')[0]


def main():
    OUT.mkdir(exist_ok=True)
    old=BASE/'eval4'
    for rel,d in read(HERE/'panel_lock.json').items():assert sha(P/rel)==d,rel
    paths=[P/'a_exploration/paired_branches_v2/cohort.json',P/'a_exploration/hssd_smoke/receipt.json',
           P/'a_hssdov_external/runtime_check/receipt.json', P/'jepa_env_v13/simulator/cohort.json',
           P/'jepa_env_v13/simulator/training_cohort.json',*sorted((P/'jepa_route_v12').rglob('*cohort.json')),
           *(BASE/k/'panel.json' for k in ('eval','eval2','eval3','eval4')),old/'eligibility.json']
    previous=set.union(*(parsed_ids(f.read_text()) for f in paths))
    historical=read(P/'jepa_hidden_v11/hssd/cohort.json')
    for k in ('train','calibration','test'):previous.update(historical[k])
    for k,vs in historical['footprints'].items():
        if k.startswith('jepa_object_search/'):previous.update(vs)
    previous.update(read(old/'locked/learned_prior.json')['train_houses'])
    # The vision module may have used the existing pretrain splits of this project.
    dev=read(P/'simulation/train_dev_v1.json')
    previous.update(dev['train'])
    previous.update(dev['dev'])
    houses=set(read(HERE/'frozen_exposure.json')['available'])
    used_families=set(map(family,previous))
    pool=sorted(s for s in houses if family(s) not in used_families)
    print('FAMILY_POOL',len(pool),pool,flush=True)
    assert pool
    for f in ('physical_panel.py','inference_worker.py'):
        if not (OUT/f).exists():shutil.copyfile(HERE/f,OUT/f)
        assert sha(OUT/f)==sha(HERE/f)
    if not (OUT/'locked').exists():
        shutil.copytree(HERE/'locked',OUT/'locked',ignore=shutil.ignore_patterns('__pycache__'))
    from physical_panel import sensor_sim
    sys.path.insert(0,str(P/'src/stopwatch'))
    from stopwatch.eval.geometry import _load_goal_sets,load_scene_instances,geodesic_distance
    import numpy as np
    goals=_load_goal_sets(DATA)['scoring_class_ids']['hssdov']
    train_targets=read(HERE/'locked/learned_prior.json')['targets']
    rows=[json.loads(l) for l in (DATA/'episode_goal_join.jsonl').read_text().splitlines()]
    panel=[];audit=[]
    for goal in GOALS:
        if goal.replace(' ','_') not in train_targets:continue
        for row in sorted((r for r in rows if r['category']==goal and r['scene'] in pool and family(r['scene']) not in used_families and r['goal_instances']>=2),key=rank):
            shard=DATA/'eval/hssdov/content'/(row['scene']+'.json.gz')
            ep=next(e for e in json.load(gzip.open(shard))['episodes'] if str(e['episode_id'])==str(row['episode_id']))
            sim=sensor_sim(ep)
            try:
                instances=load_scene_instances(DATA/'protocol/instance_viewpoints/hssd','val',ep['scene_id'])
                distances=[geodesic_distance(sim,ep['start_position'],np.float32([v['agent_state']['position'] for v in r['view_points']]))
                           for r in instances if r['class_id'] in goals[goal] and r['view_points']]
            finally:sim.close()
            conn=sum(map(math.isfinite,distances))
            audit.append(dict(scene=row['scene'],id=ep['episode_id'],goal=goal,reachable=conn,initially_close=sum(d<.25 for d in distances)))
            if conn>=2:
                panel.append(dict(scene=row['scene'],id=ep['episode_id'],goal=goal,quota=2,budget=300,shard_sha256=sha(shard)))
                used_families.add(family(row['scene']))
                break
        if len(panel)==8:break
    assert len(panel)>=1
    assert len({family(r['scene']) for r in panel})==len(panel)
    write_once(OUT/'eligibility.json',audit)
    write_once(OUT/'panel.json',panel)
    write_once(OUT/'family_exposure.json',dict(historical_scenes=sorted(previous),val_pool_after_exclusions=pool,
            all_selected_have_disjoint_numeric_prefix=all(family(r['scene']) not in {family(i) for i in previous} for r in panel),
            legacy_unrelated_exposure=read(HERE/'frozen_exposure.json')['different_project_october_place_baseline']))
    record={str(f.relative_to(P)):sha(f) for f in [*paths,OUT/'panel.json',OUT/'family_exposure.json',OUT/'eligibility.json',
        Path(__file__),HERE/'panel_lock.json',OUT/'physical_panel.py',OUT/'inference_worker.py',*(f for f in (OUT/'locked').rglob('*') if f.is_file() and f.suffix!='.pyc')]}
    write_once(OUT/'panel_lock.json',record)
    print('FROZEN_FAMILY',len(panel),sha(OUT/'panel.json'),flush=True)

if __name__=='__main__':main()
