"""Pre-evaluation independent panel. Choose at most one start per category and one per house."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path
import sys
here=Path(__file__).resolve().parent
train=here.parent/'eval3'
p=here.parents[1]
data=p/'a_hssdov_external'
GOALS=['clock','wall clock','painting','picture','cup','mug','table lamp','bed','chair','table','plant','sofa','shelf','television','stool','office chair','armchair','cabinet','pillow']
def h(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def rank(row):return hashlib.sha256(('route-generalization-confirm-v1:'+str(row['scene'])+':'+str(row['episode_id'])).encode()).hexdigest()

def main():
    prior_lock=json.loads((train/'locked_sources.json').read_text())
    for f,value in prior_lock.items():
        src=p/f
        assert h(src)==value
    for f in ('physical_panel.py','inference_worker.py'):
        dst=here/f;shutil.copyfile(train/f,dst)
        assert h(dst)==h(train/f)
    if not (here/'locked').exists():shutil.copytree(train/'locked',here/'locked')
    for src in (train/'locked').rglob('*'):
        if src.is_file() and src.suffix!='.pyc':assert h(src)==h(here/'locked'/src.relative_to(train/'locked'))
    from physical_panel import sensor_sim
    sys.path.insert(0,str(p/'src/stopwatch'))
    from stopwatch.eval.geometry import _load_goal_sets, load_scene_instances, geodesic_distance
    import numpy as np
    existing=set()
    for name in ('eval','eval2'):
        existing.update(r['scene'] for r in json.loads((here.parent/name/'panel.json').read_text()))
    used=set(existing)
    keysets=json.loads((here/'locked'/'learned_prior.json').read_text())['targets']
    cohort=json.loads((p/'jepa_env_v13/simulator/cohort.json').read_text())
    used.update(json.loads((here/'locked'/'learned_prior.json').read_text())['train_houses']);used.update(cohort['fresh']);used.update(cohort['calibration'])
    raw=[json.loads(l) for l in (data/'episode_goal_join.jsonl').read_text().splitlines()]
    goals=_load_goal_sets(data)['scoring_class_ids']['hssdov']
    panel=[];audit=[]
    # No simulator rollout, RGB observation, policy/model score or task result informs selection.
    for goal in GOALS:
        if goal.replace(' ','_') not in keysets:continue
        rows=sorted((r for r in raw if r['scene'] not in used and r['category']==goal and r['goal_instances']>=2),key=rank)
        for row in rows:
            shard=data/'eval/hssdov/content'/(row['scene']+'.json.gz')
            ep=next(e for e in json.load(gzip.open(shard))['episodes'] if str(e['episode_id'])==str(row['episode_id']))
            sim=sensor_sim(ep)
            try:
                instances=load_scene_instances(data/'protocol/instance_viewpoints/hssd','val',ep['scene_id'])
                count=0
                for r in instances:
                    if r['class_id'] in goals[goal] and r['view_points']:
                        pts=np.float32([v['agent_state']['position'] for v in r['view_points']])
                        count+=bool(np.isfinite(geodesic_distance(sim,ep['start_position'],pts)))
            finally:sim.close()
            audit.append(dict(scene=row['scene'],episode=row['episode_id'],goal=goal,start_connected_instances=count))
            if count>=2:
                panel.append(dict(scene=row['scene'],id=ep['episode_id'],goal=goal,quota=2,budget=300,shard_sha256=h(shard)))
                used.add(row['scene']);break
        if len(panel)==8:break
    (here/'eligibility.json').write_text(json.dumps(audit,indent=2))
    out=here/'panel.json'
    if out.exists():assert json.loads(out.read_text())==panel
    out.write_text(json.dumps(panel,indent=2))
    paths=[here/'eligibility.json',out,here/'accept.py',Path(__file__).resolve()]
    for rel in prior_lock:
        src=p/rel
        if (train/'locked') in src.parents or src.name in ('physical_panel.py','inference_worker.py'):
            src=here/src.relative_to(train)
        if src not in paths and (train not in src.parents):
            paths.append(src)
    record={str(f.relative_to(p)):h(f) for f in paths}
    (here/'locked_sources.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(panel),flush=True)

if __name__=='__main__':main()
