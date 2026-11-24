"""Continuous Habitat action trees for repeated, equal-budget household search."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


def seed(*parts):
    return int(hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:8],16)%2147483647


def write_json(path,data):
    pending=path.with_suffix('.pending')
    pending.write_text(json.dumps(data,separators=(',',':')))
    pending.replace(path)


def cohorts(project):
    old=json.loads((project/'jepa_hidden_v11/hssd/cohort.json').read_text())
    a=json.loads((project/'jepa_route_v12/fresh_cohort.json').read_text())
    b=json.loads((project/'jepa_route_v12/expansion/hssd/cohort.json').read_text())
    used=set(x.split('_')[0] for group in (old['protected'],old['train'],old['test'],old['calibration'],
                                           a['test'],b['train'],b['test']) for x in group)
    new=sorted(h for h in old['fresh_candidates'] if h.split('_')[0] not in used)
    assert len(new)==len(set(h.split('_')[0] for h in new))==12
    return dict(calibration=sorted(old['calibration']),fresh=new)


def visible(obs,n):
    sem,depth=obs['ids'],obs['depth']
    valid=(sem>0)&(sem<n)&np.isfinite(depth)&(depth>0)&(depth<=2.)
    counts=np.bincount(sem[valid].astype(int),minlength=n)
    return np.flatnonzero(counts>=32).tolist()


def action_options(sim,acts,number):
    """Generate candidates from geometry only. Prefer distinct displacement directions."""
    import habitat_sim as hs
    rng=np.random.default_rng(number)
    sim.pathfinder.seed(number)
    agent=sim.get_agent(0)
    initial=agent.get_state()
    start=np.asarray(initial.position)
    choices={}
    for _ in range(35):
        end=np.asarray(sim.pathfinder.get_random_navigable_point_near(start,5.)).copy()
        if not np.isfinite(end).all() or abs(start[1]-end[1])>.5:
            continue
        v=(end-start)[[0,2]]
        if np.linalg.norm(v)<1.1:
            continue
        follower=sim.make_greedy_follower(agent_id=0,goal_radius=.22,stop_key='stop',
                    forward_key=acts[0],left_key=acts[1],right_key=acts[2])
        try:
            steps=[acts.index(s) for s in follower.find_path(end) if s in acts]
        except hs.errors.GreedyFollowerError:
            continue
        if not 6<=len(steps)<=45 or steps[:24].count(0)<4:
            continue
        plan=tuple((steps+[1,1,2,2,2,2,1,1]*3)[:24])
        choices[plan]=v/np.linalg.norm(v)
        if len(choices)>=12:
            break
    if np.linalg.norm(agent.get_state().position-initial.position)>1e-5:
        raise RuntimeError('candidate planning moved the live agent')
    options=list(choices)
    rng.shuffle(options)
    selected=options[:1]
    if selected:
        while len(selected)<min(3,len(choices)):
            available=[p for p in options if p not in selected]
            selected.append(min(available,key=lambda p:max(choices[p]@choices[s] for s in selected)))
    fallback=3-len(selected)
    for fill in ([1]*24,[2]*24,([1,1,2,2]*6)):
        if len(selected)==3:
            break
        selected.append(tuple(fill))
    rng.shuffle(selected)
    return np.asarray(selected,dtype=np.uint8),fallback


def choose_targets(sim,lookup,pose,yaw,words,token):
    from collect_branches import world_state
    agent=sim.get_agent(0); visible_categories=set()
    for i in range(12):
        agent.set_state(world_state(pose,yaw+i*math.pi/6))
        ids=np.unique(sim.get_sensor_observations()['ids'])
        visible_categories.update(lookup[j] for j in ids if 0<j<len(lookup))
    groups={'multiple':[],'single':[],'absent':[]}
    for j,w in enumerate(words):
        if j in visible_categories:continue
        n=int((lookup==j).sum())
        s='multiple' if n>=2 else 'single' if n else 'absent'
        groups[s].append(dict(word=w,house_count=n,quota=min(n,2),stratum=s))
    for group in groups.values():
        group.sort(key=lambda x:seed('hidden-query-v13',token,x['word']))
    if not groups['multiple'] and not groups['single']:
        return []
    return groups['multiple'][:2]+groups['single'][:1]+groups['absent'][:1]


def scan(sim,act,n):
    agent=sim.get_agent(0);before=agent.get_state()
    frames=[];gt={}
    for i,a in enumerate((2,1,1,2),1):
        agent.act(act[a])
        if i<=3:
            obs=sim.get_sensor_observations()
            frames.append(obs['rgb'][:,:,:3].copy());gt[str(i)]=visible(obs,n)
    now=agent.get_state()
    assert np.linalg.norm(now.position-before.position)<1e-5
    assert abs(float(now.rotation.components@before.rotation.components))>1.-1e-5
    return np.asarray(frames),gt


def leg(sim,acts,plan,n):
    agent=sim.get_agent(0);frames=[];bumped=[];gt={}
    for t,a in enumerate(plan,1):
        blocked=agent.act(acts[int(a)])
        bumped.append(bool(a==0 and blocked))
        if t in (6,12,18,24):
            obs=sim.get_sensor_observations()
            frames.append(obs['rgb'][:,:,:3].copy());gt[str(t)]=visible(obs,n)
    return agent.get_state(),np.asarray(frames),np.asarray(bumped),gt


def make_tree(sim,acts,lookup,words,home,uid,dst,origin,targets):
    n=len(lookup);nodes={};fallbacks=Counter()
    def visit(path,start):
        agent=sim.get_agent(0)
        agent.set_state(start)
        views,gt_scan=scan(sim,acts,n);base=agent.get_state()
        plans,backs=action_options(sim,acts,seed('node13',home,uid,path))
        fallbacks[backs]+=1
        ends=[];future=[];collisions=[];truth=[]
        for p in plans:
            # Separate counterfactual children clone the same physically reached parent.
            agent.set_state(base)
            end,frames,bump,observed=leg(sim,acts,p,n)
            ends.append(end);future.append(frames);collisions.append(bump);truth.append(observed)
        np.savez_compressed(dst/('node_'+path+'.npz'),ctx=views,plans=plans,
                            walk_rgb=future,collided=collisions)
        nodes[path]={'scan':gt_scan,'walk':truth,'fallback_count':backs}
        if len(path)<4:
            for i,end in enumerate(ends):visit(path+str(i),end)
    visit('r',origin)
    assert len(nodes)==40
    write_json(dst/'truth.json',dict(nodes=nodes,instance_words={str(i):words[j] for i,j in enumerate(lookup) if j>=0}))
    write_json(dst/'episode.json',dict(house=home,origin=uid,targets=targets,steps_per_choice=28,
               budgets=[28,56,84,112],nodes=40,pure_turn_fallback_counts=dict(fallbacks)))


def training_queries(sim,lookup,words,xyz,yaw,dst):
    from collect_branches import world_state
    visible_at_start=set()
    for i in range(12):
        sim.get_agent(0).set_state(world_state(xyz,yaw+i*math.pi/6))
        ids=np.unique(sim.get_sensor_observations()['ids'])
        visible_at_start.update(lookup[j] for j in ids if 0<j<len(lookup))
    truth=json.loads((dst/'truth.json').read_text())
    anywhere=set()
    for node in truth['nodes'].values():
        for frame in [node['scan'],*node['walk']]:
            anywhere.update(lookup[i] for ids in frame.values() for i in ids if i<len(lookup))
    positive=[];negative=[]
    for j,w in enumerate(words):
        if j in visible_at_start:continue
        n=int((lookup==j).sum())
        row=dict(word=w,house_count=n,quota=min(2,n),stratum='multiple' if n>=2 else 'single' if n else 'absent',
                 encounterable=j in anywhere)
        (positive if j in anywhere else negative).append(row)
    negative.sort(key=lambda x:seed('train-negative13',dst.name,x['word']))
    write_json(dst/'training_targets.json',dict(targets=positive+negative[:6]))


def run(project,asset,split,house,limit=8):
    from collect_branches import ACT, WORDS, simulate, semantics, world_state
    dest=project/'jepa_env_v13/simulator/trees'/split/house; dest.mkdir(parents=True,exist_ok=True)
    if (dest/'receipt.json').is_file():return
    sim=simulate(asset,house)
    starts=[];history=[]
    try:
        lookup=semantics(sim,asset)
        for ix in range(160):
            token=seed('root13',house,ix);sim.pathfinder.seed(token)
            xyz=np.asarray(sim.pathfinder.get_random_navigable_point()).copy()
            if not np.isfinite(xyz).all() or any(np.linalg.norm(xyz-x)<1.5 for x in starts):
                continue
            yaw=float(np.random.default_rng(token).uniform(-math.pi,math.pi));uid='%s-%03d'%(house,ix)
            pose=world_state(xyz,yaw);sim.get_agent(0).set_state(pose)
            _,backs=action_options(sim,ACT,seed('node13',house,uid,'r'))
            if backs:continue
            targets=choose_targets(sim,lookup,xyz,yaw,WORDS,uid)
            if not targets:continue
            dst=dest/uid;dst.mkdir(exist_ok=True);starts.append(xyz)
            history.append(dict(origin=uid,position=xyz.tolist(),yaw=yaw))
            if not (dst/'episode.json').exists():
                make_tree(sim,ACT,lookup,WORDS,house,uid,dst,pose,targets)
            if split=='train':
                training_queries(sim,lookup,WORDS,xyz,yaw,dst)
            if len(starts)==limit:break
    finally:sim.close()
    write_json(dest/'private_states.json',history)
    write_json(dest/'receipt.json',dict(house=house,origins=len(starts),attempts=ix+1))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('project',type=Path);p.add_argument('asset',type=Path)
    p.add_argument('split');p.add_argument('house')
    p.add_argument('--origins',type=int,default=8)
    a=p.parse_args();sys.path.insert(0,str(a.project/'jepa_route_v12/expansion/hssd'))
    run(a.project,a.asset,a.split,a.house,a.origins)
