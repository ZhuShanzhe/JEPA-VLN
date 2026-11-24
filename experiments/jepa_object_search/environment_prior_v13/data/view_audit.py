"""Blind room-reading and four-tap lookahead audit on a predetermined trajectory."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from rooms import Atlas,ROOMS


def capture(project,assets):
    from room_teacher import view_label
    d=project/'jepa_env_v13'
    sys.path[:0]=[str(d/'simulator'),str(project/'jepa_route_v12/hssd')]
    from tree import seed
    from collect_branches import ACT,simulate,semantics,world_state
    dst=d/'data/diagnostic_private';dst.mkdir(exist_ok=True)
    for house in json.loads((d/'simulator/cohort.json').read_text())['fresh']:
        home=d/'simulator/trees/fresh'/house
        item=min(json.loads((home/'private_states.json').read_text()),key=lambda x:x['origin'])
        origin=item['origin']; folder=home/origin
        atlas=Atlas(assets,house);sim=simulate(assets,house);semantics(sim,assets)
        path='r'; records=[]
        try:
            agent=sim.get_agent(0);agent.set_state(world_state(item['position'],item['yaw']))
            def view(reference):
                obs=sim.get_sensor_observations()
                if not np.array_equal(reference,obs['rgb'][:,:,:3]):
                    raise ValueError('independent RGB replay mismatch')
                return view_label(atlas,agent.get_state().sensor_states['depth'],obs)
            for k in range(4):
                choice=seed('independent_replay13',house,origin,k)%3
                with np.load(folder/('node_'+path+'.npz')) as z:
                    labels=[]
                    for i,command in enumerate((2,1,1,2)):
                        agent.act(ACT[command])
                        if i<3:labels.append(view(z['ctx'][i]))
                    for t,command in enumerate(z['plans'][choice],1):
                        agent.act(ACT[int(command)])
                        if t%6==0:labels.append(view(z['walk_rgb'][choice,t//6-1]))
                records.append(dict(node=path,choice=int(choice),labels=np.stack(labels).tolist()))
                path+=str(choice)
        finally:
            sim.close()
        (dst/(house+'.json')).write_text(json.dumps(dict(origin=origin,nodes=records)))
    print('calibration-free physical diagnostic complete',flush=True)


def copy_index(plan):
    yaw=np.cumsum((plan==1).astype(float)-(plan==2))*np.pi/6
    dx=np.cumsum((plan==0)*.25*np.cos(yaw));dz=np.cumsum((plan==0)*.25*np.sin(yaw))
    bearing=np.where(np.hypot(dx,dz)>.05,np.arctan2(dz,dx),yaw)
    return np.cos(bearing[5::6,None]-np.deg2rad([-30,0,30])).argmax(1)


def metric(truth,pred):
    valid=truth.sum(-1)>0
    truth=truth[valid];pred=pred[valid]
    if not len(truth):return dict(n=0)
    y=truth.argmax(-1);effect=pred.argmax(-1)
    cf=np.zeros((len(ROOMS),len(ROOMS)),int);np.add.at(cf,(y,effect),1)
    return dict(n=len(y),accuracy=float((effect==y).mean()),
                brier=float(np.mean(((truth-pred)**2).sum(-1))),confusion=cf.tolist())


def score(project):
    d=project/'jepa_env_v13'
    sys.path.insert(0,str(d/'models'));from env_routes import EnvironmentRoutes
    tables={}; flips=[]; stat={}
    for sd in (11,29,47):
        model=EnvironmentRoutes(project,sd);buffer={};fixed=0;changed=0
        def add(name,truth,pred,house):
            buffer.setdefault(name,[]).append((house,truth,pred))
        for f in sorted((d/'data/diagnostic_private').glob('*.json')):
            house=f.stem;records=json.loads(f.read_text())
            folder=d/'simulator/trees/fresh'/house/records['origin']
            goal=json.loads((folder/'episode.json').read_text())['targets'][0]['word']
            for record in records['nodes']:
                y=np.array(record['labels']);choice=record['choice']
                with np.load(folder/('node_'+record['node']+'.npz')) as z:
                    route=model.live(z['ctx'],goal,z['plans'])
                    now=model.observed(np.concatenate([z['ctx'],z['walk_rgb'][choice]]))
                    indexes=copy_index(z['plans'][choice])
                start=y[indexes]; end=y[3:]
                add('current_rgb',y,now,house)
                seen=(start.sum(-1)>0)&(end.sum(-1)>0)
                diff=start.argmax(-1)!=end.argmax(-1)
                fixed+=int(seen.sum());changed+=int((seen&diff).sum())
                for arm,forecast in [('jepa',route['env_steps_jepa'][choice]),
                                     ('direct',route['env_steps_direct'][choice]),('copy',now[indexes])]:
                    add(arm,end,forecast,house)
                    for condition,mask in [('change',seen&diff),('same',seen&~diff)]:
                        add(arm+'_'+condition,end[mask],forecast[mask],house)
        for name,rows in buffer.items():
            truths=np.concatenate([x[1] for x in rows]);estimates=np.concatenate([x[2] for x in rows])
            trial=metric(truths,estimates)
            by_house=[]
            for h in sorted(set(x[0] for x in rows)):
                g=[x for x in rows if x[0]==h]
                m=metric(np.concatenate([x[1] for x in g]),np.concatenate([x[2] for x in g]))
                if m['n']:by_house.append((m['accuracy'],m['brier']))
            trial['house_macro_accuracy_brier']=np.mean(by_house,axis=0).tolist() if by_house else None
            tables.setdefault(name,[]).append(trial)
        flips.append([fixed,changed])
        del model
    assert len(set(tuple(x) for x in flips))==1
    for name,runs in tables.items():
        cf=np.mean([v['confusion'] for v in runs],0)
        stat[name]=dict(n=runs[0]['n'],seed_accuracy=[v['accuracy'] for v in runs],
                        mean_brier=float(np.mean([v['brier'] for v in runs])),
                        house_macro_mean_accuracy_brier=np.mean([v['house_macro_accuracy_brier'] for v in runs],0).tolist(),
                        confusion_mean_over_seeds=cf.tolist())
    out=dict(room_names=ROOMS,houses=12,nodes=48,seeds=[11,29,47],transition_valid=flips[0][0],
         transition_changed=flips[0][1],metrics=stat)
    (d/'data/view_audit.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('project',type=Path);p.add_argument('assets',type=Path)
    p.add_argument('task',choices=('capture','score'));a=p.parse_args()
    if a.task=='capture':capture(a.project,a.assets)
    else:score(a.project)
