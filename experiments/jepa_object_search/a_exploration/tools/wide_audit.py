"""Independent validation and indexing of the frozen v2 study cohort."""
import argparse, collections, hashlib, json, pathlib
import numpy as np
root=pathlib.Path(__file__).resolve().parents[2];base=root/'a_exploration/paired_branches_v2'
cohort=json.loads((base/'cohort.json').read_text())
parts=root/'simulation/train_dev_v1.json';split=json.loads(parts.read_text())
assert hashlib.sha256(parts.read_bytes()).hexdigest()==cohort['split_sha256']
parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');arg=parser.parse_args()
allrows=[];counts=collections.defaultdict(collections.Counter);scene_counts=collections.defaultdict(collections.Counter)
house=set();nfiles=0
for j in cohort['jobs']:
    s,h,tag=j['split'],j['scene'],j['target'];assert h in split[s]
    folder=base/s/h/f"seed{j['seed']}-steps{j['budget']}-{tag}"
    meta=sorted(folder.glob('start*/manifest.json'))
    if not meta and arg.partial:continue
    if not arg.partial:assert len(meta)==j['samples'],(h,len(meta))
    for m in meta:
        data=json.loads(m.read_text());n=data['frontier_count'];k=data['action_budget'];assert n>=2 and k==j['budget']
        item=m.parent.relative_to(base).as_posix()
        labels=json.loads((m.parent/'teacher/labels.json').read_text())
        assert s==data['split'] and h==data['source_scene'] and tag==labels['synset'] and j['seed']==data['seed']
        assert data['split_sha256']==cohort['split_sha256']
        with np.load(m.parent/'context.npz') as ctx:
            assert ctx['rgb'].shape==(12,256,256,3) and np.isfinite(ctx['depth_m']).all()
            assert ctx['frontiers_xy'].shape==(n,2) and ctx['planned_actions'].shape==(n,k)
            assert int(ctx['planned_actions'].max())<3
        if not arg.partial:
            with np.load(m.parent/'features_vjepa21_b384.npz') as z:
                assert z['context_embedding'].shape==(768,) and z['future_targets'].shape==(n,768)
                assert np.isfinite(z['context_embedding']).all() and np.isfinite(z['future_targets']).all()
        assert len(labels['branch_labels'])==n
        prefix=np.asarray(labels['prefix_pixel_counts'])
        values=[]
        for i,row in enumerate(labels['branch_labels']):
            with np.load(m.parent/f'branch{i:03d}.npz') as b, np.load(m.parent/'teacher'/f'branch{i:03d}.npz') as t:
                assert b['rgb'].shape==(k,256,256,3) and b['depth_m'].shape==(k,256,256) and np.isfinite(b['depth_m']).all()
                assert b['relative_pose_xyyaw'].shape==(k,3) and b['actions'].shape==(k,) and b['actions'].max()<3
                threshold=labels['visible_pixel_threshold'];pix=t['target_pixel_counts']
                assert pix.shape==(k,len(t['instance_ids']))
                new=t['instance_ids'][(prefix.max(0)<threshold)&(pix.max(0)>=threshold)].tolist()
                assert row['newly_visible_ids']==new
                values.append(bool(new));nfiles+=1
        record=dict(id=item,scene=h,split=s,target=tag,branches=n,positive=sum(values),future_frames=k*n,choice_changes_novel_outcome=(min(values)!=max(values)))
        stats=dict(states=1,branches=n,positive_branches=sum(values),future_frames=k*n,discriminating_states=int(record['choice_changes_novel_outcome']))
        counts[s].update(stats);counts[s+'/'+tag].update(stats);scene_counts[s+'/'+h].update(stats)
        allrows.append(record);house.add((s,h))
suffix='_partial' if arg.partial else ''
(base/f'index{suffix}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in allrows))
summary=dict(rows=len(allrows),complete=not arg.partial,cohort_sha256=hashlib.sha256((base/'cohort.json').read_bytes()).hexdigest(),
             counts=counts,houses={s:sum(x==s for x,_ in house) for s in ('train','dev')},by_house=scene_counts,
             frozen_feature_shapes_checked=not arg.partial)
(base/f'audit{suffix}.json').write_text(json.dumps(summary,indent=2))
print(summary['houses'],counts['train'],counts['dev'],flush=True)
