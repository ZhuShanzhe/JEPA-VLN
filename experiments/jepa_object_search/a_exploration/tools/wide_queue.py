"""Freeze the expanded cohort using only scene IDs and asset vocabulary, then collect."""
import collections, concurrent.futures, csv, hashlib, json, os, pathlib, subprocess, sys,time

root=pathlib.Path(__file__).resolve().parents[2];area=root/'a_exploration'
base=area/'paired_branches_v2';base.mkdir(exist_ok=True)
logs=base/'logs';logs.mkdir(exist_ok=True)
sim='/root/autodl-tmp/jepa_instance_bench_20261016/runtime/jev_habitat_workspace/.pixi/envs/default/bin/python'
raw=pathlib.Path('/root/autodl-tmp/jepa_vln_data_20261014/raw/hssd-hab')
partsfile=root/'simulation/train_dev_v1.json';parts=json.loads(partsfile.read_text())
tags=['chair.n.01','bed.n.01','potted_plant.n.01','toilet.n.02','television_receiver.n.01','sofa.n.01']
seed='A-branch-v2-20261114'
def key(txt):return hashlib.sha256((seed+':'+txt).encode()).hexdigest()
asset_meta={row['id']:row for row in csv.DictReader((raw/'semantics/objects.csv').open())}
excluded={'103997478_171030525','102344049'}
jobs=[];selection={}
for split,num in [('train',12),('dev',4)]:
    remaining=[]
    for h in sorted((set(parts[split])-excluded),key=key):
        sf=raw/f'scenes/{h}.scene_instance.json'
        # The known compatibility overlay affects one scene. Keep original visuals in this series.
        if h=='108736611_177263226' or not sf.exists():continue
        objs=json.loads(sf.read_text())['object_instances']
        available=collections.Counter()
        for ob in objs:
            info=asset_meta.get(ob['template_name'],{})
            names={n.strip() for col in ('wnsynsetkey','main_wnsynsetkey') for n in info.get(col,'').split(',')}
            for t in set(tags)&names:available[t]+=1
        if available:remaining.append((h,available))
    assert len(remaining)>=num
    sel=remaining[:num];selection[split]=[x[0] for x in sel]
    use=collections.Counter()
    for house_idx,(h,av) in enumerate(sel):
        local=collections.Counter()
        for group in range(3):
            t=min(av,key=lambda tag:(local[tag],use[tag],key(f'{split}/{h}/{group}/{tag}')))
            use[t]+=1;local[t]+=1
            js=int(key(f'{split}/{h}/{group}')[:7],16)
            jobs.append(dict(split=split,scene=h,target=t,seed=js,samples=4,budget=24,attempts=80,asset_instances=av[t]))
manifest=dict(seed=seed,split_sha256=hashlib.sha256(partsfile.read_bytes()).hexdigest(),series=base.name,
    selection=selection,jobs=jobs,rule='hash-ordered non-pilot homes with >=1 authored class; three independent starts groups per house; choose present class balancing house and split counts; all candidates retained, all zero-signal outcomes retained',
    filters='exclude absent scene files and known-one-scene renderer material incompatibility; navmesh valid start and at least two observed frontiers; no task-outcome filtering',classes=tags)
mf=base/'cohort.json';body=json.dumps(manifest,indent=2)
if mf.exists():assert mf.read_text()==body
else:mf.write_text(body)
print('FROZEN',len(jobs),selection,flush=True)
env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MAGNUM_LOG':'quiet','HABITAT_SIM_LOG':'quiet','PYTHONUNBUFFERED':'1'}
def run(job):
    name=f"{job['split']}-{job['scene']}-{job['seed']}"
    done=logs/(name+'.json')
    if done.exists() and json.loads(done.read_text())['exit']==0:return 0
    cmd=[sim,str(area/'tools/collect_branches.py'),'--series',base.name,'--split',job['split'],'--scene',job['scene'],
        '--seed',str(job['seed']),'--target-synset',job['target'],'--samples','4','--budget','24','--attempts','80']
    started=time.time()
    with (logs/(name+f'-{int(started)}.log')).open('wb') as f:
        result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    done.write_text(json.dumps(dict(job=job,exit=result.returncode,wall_seconds=time.time()-started)))
    print('EXIT',result.returncode,name,flush=True)
    return result.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    statuses=list(pool.map(run,jobs))
print('END',dict(collections.Counter(statuses)),flush=True)
sys.exit(int(any(statuses)))
