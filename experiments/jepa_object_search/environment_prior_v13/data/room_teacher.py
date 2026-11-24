"""Replay training branches to label occupied rooms and masked, visible room geometry."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from rooms import Atlas,ROOMS
from train_prior import source


def view_label(atlas,state,obs):
    # Depth is optical-axis distance for the Habitat pinhole depth sensor.
    z=obs['depth'];ids=obs['ids']
    h,w=z.shape
    v,u=np.meshgrid(np.arange(6,h,12),np.arange(6,w,12),indexing='ij')
    d=z[v,u];ok=np.isfinite(d)&(d>.35)&(d<8.)&(ids[v,u]==0)
    d=d[ok]-.12
    if not len(d):
        return np.zeros(len(ROOMS))
    focal=.5*w # horizontal field of view 90 degrees
    xyz=np.stack([(u[ok]+.5-w*.5)*d/focal,-(v[ok]+.5-h*.5)*d/focal,-d],axis=-1)
    q=state.rotation
    xyz=xyz+2*np.cross(q.imag,np.cross(q.imag,xyz)+q.real*xyz)
    rooms=atlas.locate(xyz+np.asarray(state.position))
    hits=np.bincount(rooms[rooms>=0],minlength=len(ROOMS)).astype(float)
    return hits/max(1,hits.sum())


def house_replay(project,assets,split,house):
    sys.path.insert(0,str(project/'jepa_route_v12/hssd'))
    from collect_branches import ACT,simulate,semantics,world_state
    root=project/'jepa_env_v13/data';where=source(project)[split][0]/house
    out=root/'labels'/split;out.mkdir(parents=True,exist_ok=True)
    audit=root/'audits'/split; audit.mkdir(parents=True,exist_ok=True)
    receipt=audit/(house+'.json')
    if receipt.exists():
        return
    atlas=Atlas(assets,house); sim=simulate(assets,house);semantics(sim,assets)
    agent=sim.get_agent(0)
    summary=dict(origins=0,total_taps=0,known_pose=0,known_view=0,
                 assignment_changes=0,different_end_views=0,replayed_frames=0)
    try:
        for e in json.loads((where/'private_replay.json').read_text()):
            origin=e['origin']
            with np.load(where/(origin+'.npz')) as old:
                actions=old['actions']; expected=old['future']; start_rgb=old['ctx']
            room=np.full((3,4),-1,int); views=np.zeros((3,4,len(ROOMS)))
            base=world_state(e['position'],e['yaw'])
            start_room=int(atlas.locate([base.position])[0])
            start_view=[]
            for i,degrees in enumerate((-30,0,30)):
                agent.set_state(world_state(e['position'],e['yaw']+np.deg2rad(degrees)))
                obs=sim.get_sensor_observations()
                if not np.array_equal(obs['rgb'][:,:,:3],start_rgb[i]):
                    raise ValueError('recorded start mismatch: '+origin)
                start_view.append(view_label(atlas,agent.get_state().sensor_states['depth'],obs))
            for i,plan in enumerate(actions):
                agent.set_state(base)
                for t,action in enumerate(plan,1):
                    agent.act(ACT[int(action)])
                    if t%6:
                        continue
                    k=t//6-1; s=agent.get_state()
                    obs=sim.get_sensor_observations()
                    if not np.array_equal(obs['rgb'][:,:,:3],expected[i,k]):
                        raise ValueError('recorded RGB replay mismatch: '+origin)
                    room[i,k]=atlas.locate([s.position])[0]
                    views[i,k]=view_label(atlas,s.sensor_states['depth'],obs)
            summary['origins']+=1;summary['total_taps']+=12;summary['replayed_frames']+=12
            summary['known_pose']+=int((room>=0).sum())
            summary['known_view']+=int((views.sum(-1)>0).sum())
            summary['assignment_changes']+=int(np.any((room!=start_room)&(room>=0)) and start_room>=0)
            ids=views[:,-1].argmax(-1)[views[:,-1].sum(-1)>0]
            summary['different_end_views']+=int(len(set(ids))>1)
            f=out/(origin+'.npz');temp=f.with_suffix('.tmp.npz')
            np.savez_compressed(temp,room_idx=room,view_room=views.astype('float32'),start_room=start_room,
                                start_view_room=np.asarray(start_view,dtype='float32'))
            temp.replace(f)
    finally:
        sim.close()
    receipt.write_text(json.dumps(summary))


def main(a):
    if a.house:
        house_replay(a.project,a.assets,a.split,a.house)
        return
    jobs=[(s,h) for s,(base,hs) in source(a.project).items() for h in hs]
    def run(job):
        s,h=job
        p=subprocess.run([sys.executable,'-Bu',__file__,str(a.project),str(a.assets),'--split',s,'--house',h],
                         capture_output=True,text=True)
        if p.returncode:
            raise RuntimeError(str(job)+' '+p.stderr[-1500:])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run,jobs))
    out={}
    for s in source(a.project):
        hs=[json.loads(f.read_text()) for f in (a.project/'jepa_env_v13/data/audits'/s).glob('*.json')]
        out[s]=dict(houses=len(hs),**{k:sum(r[k] for r in hs) for k in hs[0]})
    (a.project/'jepa_env_v13/data/room_support.json').write_text(json.dumps(out,indent=2))
    print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('project',type=Path);p.add_argument('assets',type=Path)
    p.add_argument('--split');p.add_argument('--house')
    main(p.parse_args())
