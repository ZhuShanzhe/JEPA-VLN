"""Collect unlabelled RGB-D transitions and isolated semantic teacher labels on train/dev only."""
import argparse, csv, hashlib, json, math, os, re, sys, tempfile, types
from pathlib import Path
import numpy as np
import habitat_sim as hs
import magnum as mn
from habitat_sim.utils.common import quat_to_coeffs

root=Path(__file__).resolve().parents[2]
area=root/'a_exploration'
raw=Path('/root/autodl-tmp/jepa_vln_data_20261014/raw/hssd-hab')
partsfile=root/'simulation/train_dev_v1.json'
parts=json.loads(partsfile.read_text())
p=argparse.ArgumentParser()
p.add_argument('--split',choices=['train','dev'],required=True)
p.add_argument('--scene',required=True)
p.add_argument('--seed',type=int,default=7714)
p.add_argument('--samples',type=int,default=2)
p.add_argument('--budget',type=int,default=8)
p.add_argument('--attempts',type=int,default=25)
p.add_argument('--target-synset',default='chair.n.01')
p.add_argument('--series',choices=['paired_branches_v1','paired_branches_v2'],default='paired_branches_v1')
a=p.parse_args()
assert a.scene in parts[a.split] and a.scene not in parts[{'train':'dev','dev':'train'}[a.split]]
assert 1<=a.samples<=a.attempts and a.budget>0
assert re.fullmatch('[A-Za-z0-9._-]+',a.target_synset)
manifest_digest=hashlib.sha256(partsfile.read_bytes()).hexdigest()
sys.path[:0]=[str(root/'src/vlfm'),str(area/'frontier_exploration')]
pkg=types.ModuleType('frontier_exploration');pkg.__path__=[str(area/'frontier_exploration/frontier_exploration')]
sys.modules['frontier_exploration']=pkg
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix

config=hs.SimulatorConfiguration()
config.scene_dataset_config_file=str(raw/'hssd-hab.scene_dataset_config.json')
config.scene_id=a.scene;config.enable_physics=True;config.allow_sliding=False
agent=hs.agent.AgentConfiguration();agent.radius=.18;agent.height=1.4
cams=[]
for name,kind in [('rgb',hs.SensorType.COLOR),('depth',hs.SensorType.DEPTH),('semantic',hs.SensorType.SEMANTIC)]:
    c=hs.CameraSensorSpec();c.uuid=name;c.sensor_type=kind;c.resolution=[256,256]
    c.position=mn.Vector3(0,1.25,0);c.hfov=mn.Deg(90);cams.append(c)
agent.sensor_specifications=cams
actions=['move_forward','turn_left','turn_right']
agent.action_space={k:hs.agent.ActionSpec(k,hs.agent.ActuationSpec(amount=v)) for k,v in zip(actions,[.25,30.,30.])}
# Pin original semantics for the entire scene. Only the offline target writer reads these IDs.
ids={row['id']:row for row in csv.DictReader((raw/'semantics/objects.csv').open())}
folder=area/a.series/a.split/a.scene/f'seed{a.seed}-steps{a.budget}-{a.target_synset}'
folder.mkdir(parents=True,exist_ok=True)
def save_counts(observation, masks):
    arr=observation['semantic']
    return [int(np.count_nonzero(arr==sid)) for sid in masks]
def actor_pose(ag):
    state=ag.get_state()
    return [*map(float,state.position),*map(float,quat_to_coeffs(state.rotation))]

def plan_at_decision(target):
    """Nominal no-collision turn/forward proposal available before a branch is executed."""
    xy=np.zeros(2);yaw=0.;seq=[]
    for _ in range(a.budget):
        rem=target-xy
        diff=math.atan2(math.sin(math.atan2(rem[1],rem[0])-yaw),math.cos(math.atan2(rem[1],rem[0])-yaw))
        ac=1 if diff>math.pi/12 else (2 if diff< -math.pi/12 else 0)
        seq.append(ac)
        if ac==0:xy+=.25*np.array([math.cos(yaw),math.sin(yaw)])
        else:yaw+=math.pi/6 if ac==1 else -math.pi/6
    return np.array(seq,dtype=np.uint8)
sim=hs.Simulator(hs.Configuration(config,[agent]))
try:
    sim.seed(a.seed);nav=hs.NavMeshSettings();nav.set_defaults();nav.agent_height=1.4;nav.agent_radius=.18;nav.include_static_objects=True
    assert sim.recompute_navmesh(sim.pathfinder,nav)
    matches={};matched_asset_count=0
    for sid,(name,obj) in enumerate(sorted(sim.get_rigid_object_manager().get_objects_by_handle_substring('').items()),1):
        obj.semantic_id=sid
        # Habitat instance handles can contain a trailing colon and per-template digits.
        # Test exact metadata asset, then the canonical template handle stem.
        rawname=str(name).split(':')[0].rstrip('_')
        handle=str(obj.creation_attributes.handle)
        asset=rawname if rawname in ids else Path(handle).name.split('.')[0]
        if asset not in ids:continue
        matched_asset_count+=1
        keys=['wnsynsetkey','main_wnsynsetkey'] if a.series=='paired_branches_v2' else ['wnsynsetkey']
        names={n.strip() for key in keys for n in ids[asset][key].split(',')}
        if a.target_synset in names: matches[sid]=dict(handle=name,asset=asset)
    assert matched_asset_count>0,'unresolved HSSD object metadata'
    masks=sorted(matches)
    if not masks:raise ValueError(f'No resolved objects for requested HSSD synset {a.target_synset} in {a.scene}')
    del obj
    ag=sim.get_agent(0);saved=[]
    for trial in range(a.attempts):
        state=hs.AgentState();state.position=sim.pathfinder.get_random_navigable_point();ag.set_state(state)
        curr=sim.get_sensor_observations();grid=ObstacleMap(.1,1.2,.18)
        rgb=[];depth=[];prefix_counts=[]
        for step in range(12):
            d=curr['depth'].astype('float32');depth.append(d.copy());rgb.append(curr['rgb'][:,:,:3].copy())
            prefix_counts.append(save_counts(curr,masks))
            transform=xyz_yaw_to_tf_matrix(np.array([0.,0.,1.25]),math.pi*step/6)
            grid.update_map(np.clip(d/5.,0,1),transform,0.,5.,128.,128.,math.pi/2)
            curr=sim.step('turn_left')
        front=np.asarray(grid.frontiers,dtype=np.float32).reshape(-1,2)
        if len(front)<2:continue
        location=folder/f'start{trial:04d}'
        if location.is_dir():
            chk=json.loads((location/'manifest.json').read_text())
            assert chk['split_sha256']==manifest_digest and chk['frontier_count']==len(front)
            saved.append(str(location))
            if len(saved)>=a.samples:break
            continue
        start=ag.get_state();frozen=actor_pose(ag);labels=[]
        prefix_counts=np.array(prefix_counts,dtype=np.int32);known=prefix_counts.max(axis=0)>=64
        with tempfile.TemporaryDirectory(dir=folder) as td:
            dest=Path(td)/'state';(dest/'teacher').mkdir(parents=True)
            ctx=dict(rgb=rgb,depth_m=depth,frontiers_xy=front)
            if a.series=='paired_branches_v2':
                ctx['planned_actions']=np.stack([plan_at_decision(target) for target in front])
            np.savez_compressed(dest/'context.npz',**ctx)
            for i,target in enumerate(front):
                ag.set_state(start);assert np.allclose(actor_pose(ag),frozen,atol=1e-6)
                start_pos=np.array(frozen[:3]);yaw=0.;seq=[];rgbs=[];ds=[];counts=[];relative_pose=[];collisions=[]
                for _ in range(a.budget):
                    pos=np.array(ag.get_state().position)
                    xy=np.array([-(pos[2]-start_pos[2]),-(pos[0]-start_pos[0])]);rem=target-xy
                    delta=math.atan2(math.sin(math.atan2(rem[1],rem[0])-yaw),math.cos(math.atan2(rem[1],rem[0])-yaw))
                    ac=1 if delta>math.pi/12 else (2 if delta< -math.pi/12 else 0)
                    yaw+=math.pi/6 if ac==1 else (-math.pi/6 if ac==2 else 0.)
                    curr=sim.step(actions[ac]);seq.append(ac);rgbs.append(curr['rgb'][:,:,:3].copy());ds.append(curr['depth'].copy())
                    counts.append(save_counts(curr,masks));pos2=ag.get_state().position
                    relative_pose.append([-(pos2[2]-start_pos[2]),-(pos2[0]-start_pos[0]),yaw]);collisions.append(bool(curr['collided']))
                cc=np.array(counts,np.int32)
                labels.append(dict(index=i,prefix_known_instance_ids=np.array(masks)[known].tolist(),
                      newly_visible_ids=np.array(masks)[(~known)&(cc.max(axis=0)>=64)].tolist(),
                      collided_steps=int(sum(collisions))))
                np.savez_compressed(dest/f'branch{i:03d}.npz',actions=np.array(seq,np.uint8),rgb=rgbs,
                     depth_m=ds,relative_pose_xyyaw=relative_pose,collided=collisions)
                np.savez_compressed(dest/'teacher'/f'branch{i:03d}.npz',instance_ids=masks,target_pixel_counts=cc)
            (dest/'teacher'/'labels.json').write_text(json.dumps(dict(
                synset=a.target_synset,visible_pixel_threshold=64,instance_map=matches,object_metadata_matches=matched_asset_count,
                prefix_pixel_counts=prefix_counts.tolist(),branch_labels=labels),indent=2))
            metadata=dict(scope='train/dev research data, no benchmark score',split=a.split,split_sha256=manifest_digest,
                source_scene=a.scene,seed=a.seed,start_idx=trial,frozen_pose_xyz_xyzw=frozen,frontier_count=len(front),
                action_budget=a.budget,action_vocab=actions,branch_controller='position-only heading follower',
                history='12 observed turn-left RGB-D frames; labels isolated under teacher/',
                novel_definition='not visible in observed prefix at threshold, not an earlier report',future='all proposed branches, including ineffective outcomes')
            (dest/'manifest.json').write_text(json.dumps(metadata,indent=2))
            os.rename(dest,location)
        saved.append(str(location));print('saved',location,len(front),flush=True)
        if len(saved)>=a.samples:break
    assert len(saved)==a.samples,f'Only {len(saved)} eligible states in {a.attempts} trials'
    print('SUCCESS',len(saved),flush=True)
finally:sim.close()
