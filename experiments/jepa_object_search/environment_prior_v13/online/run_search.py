"""Choose four consecutive real branches without consulting simulation outcomes."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
from scene_memory import SceneMemory

SEEDS=(11,29,47)
# Room prior, scene evidence, exploration bonus, dedicated second-round inspection.
SETTINGS={'model_only':(0.,0.,0.,False),'weak':(.12,1.,.04,False),
          'balanced':(.25,2.,.09,False),'scout':(.25,2.,.09,True),
          'scout_strong':(.4,3.,.14,True)}


def pick_route(base,room,prior,terms,turn,config):
    alpha,gain,explore,scout=config
    affinity=room@prior
    scores=SceneMemory.apply(base+alpha*(affinity-affinity.mean()),room,terms,
                             gain=gain,exploration=explore)
    pick=int(np.argmax(scores));scout_changed=False
    if scout and turn==1:
        plausible=scores>=scores.max()-.20
        uncertainty=room@terms[2]
        value=terms[3]*(.35+.65*uncertainty)+.12*(scores-scores.max())
        alter=int(np.argmax(np.where(plausible,value,-np.inf)))
        scout_changed=alter!=pick;pick=alter
    return pick,scout_changed


class Junctions:
    def __init__(self,bank,q,old,project,calibrator):
        self.bank=bank;self.q=q;self.index={p:i for i,p in enumerate(bank['paths'].tolist())}
        self.mem0=SceneMemory(old.remem(project),calibrator);self.states={}; self.terms={}

    def current(self,path):
        if path in self.terms:return self.terms[path]
        bank=self.bank
        if path=='r':
            memory=self.mem0
        else:
            parent=path[:-1]; pi=self.index[parent]; branch=int(path[-1])
            self.current(parent)
            memory=copy.deepcopy(self.states[parent])
            for t,(act,blocked) in enumerate(zip(bank['plans'][pi,branch],bank['blocked'][pi,branch]),1):
                obs=3+4*branch+t//6-1 if t%6==0 else None
                memory.observe(act,blocked,None if obs is None else bank['sensed'][pi,obs],
                               None if obs is None else float(bank['cue'][pi,obs,self.q]))
        i=self.index[path]
        for t,act in enumerate((2,1,1,2)):
            memory.observe(act,False,bank['sensed'][i,t] if t<3 else None,
                           float(bank['cue'][i,t,self.q]) if t<3 else None)
        self.states[path]=memory
        self.terms[path]=memory.components(bank['plans'][i])
        return self.terms[path]

    def choices(self,seed,method,config,horizon=False,layout=False,transition=False):
        banks=self.bank;q=self.q
        path='r'; steps=[];scout_changed=False
        for turn in range(4):
            i=self.index[path];terms=self.current(path)
            base=banks[method if method in ('v11','direct','jepa') else 'v11'][seed,i,q].astype(float)
            room=banks['env_pred_'+(method if method in ('direct','jepa') else 'copy')][seed,i,q]
            prior=banks['prior'][seed,i,q]
            if horizon:
                base+=banks['horizon_'+method][seed,i,q]
            if layout:
                base+=banks['adjust_'+method][seed,i,q]
            if transition:
                room=banks['transition'][seed,i,q]
            pick,changed=pick_route(base,room,prior,terms,turn,config)
            scout_changed |= changed
            path+=str(pick);steps.append(pick)
        return steps,scout_changed


def assess(tree,steps,word,quota):
    # Verdicts are read after a full causal trajectory has been committed.
    seen=set();first_step=second_step=None;path='r';records=[]
    def remember(ids,when):
        nonlocal first_step,second_step
        for inst in ids:
            if tree['instance_words'].get(str(inst))==word:seen.add(inst)
        if seen and first_step is None:first_step=when
        if len(seen)>=2 and second_step is None:second_step=when
    for k,choice in enumerate(steps):
        ev=tree['nodes'][path]
        for d,ids in sorted(ev['scan'].items(),key=lambda v:int(v[0])):remember(ids,k*28+int(d))
        for d,ids in sorted(ev['walk'][choice].items(),key=lambda v:int(v[0])):remember(ids,k*28+4+int(d))
        records.append(len(seen));path+=str(choice)
    return dict(found=records,first_step=first_step,second_step=second_step)


def policies(split,path,stage):
    if split=='calibration':
        return {f'jepa_{setting}':('jepa',cfg) for setting,cfg in SETTINGS.items()}
    sel=json.loads(path.read_text())['setting'];cfg=SETTINGS[sel];a,g,e,s=cfg
    if stage:
        variants={'v11_baseline':('v11',SETTINGS['model_only']),'jepa_full':('jepa',cfg),
                  'direct_with_update':('direct',cfg),'jepa_horizon':('jepa',cfg),'direct_horizon':('direct',cfg)}
        if stage=='layout':
            variants.update(jepa_horizon_layout=('jepa',cfg),jepa_layout=('jepa',cfg))
        if stage=='transition':
            variants.update(jepa_horizon_transition=('jepa',cfg), jepa_transition=('jepa',cfg))
        return variants
    return {'v11_baseline':('v11',SETTINGS['model_only']),'direct_with_update':('direct',cfg),
            'jepa_full':('jepa',cfg),'jepa_prior_only':('jepa',(a,0.,0.,False)),
            'jepa_without_scene_evidence':('jepa',(a,0.,e,s)),
            'jepa_model_only':('jepa',SETTINGS['model_only']),
            'v11_departure_prior':('copy',(a,0.,0.,False)),
            'jepa_dedicated_scout':('jepa',SETTINGS['scout'])}


def main(project,split,stage):
    root=project/'jepa_env_v13';online=root/'online'
    sys.path.insert(0,str(project/'jepa_hidden_v11/models'))
    import train_compare as old
    cue=json.loads((online/'cue_calibration.json').read_text())
    configs=policies(split,online/'online_selection.json',stage)
    logs=[]
    for filename in sorted((online/'scored'/split).glob('*/*.npz')):
        house=filename.parent.name;origin=filename.stem;raw=root/'simulator/trees'/split/house/origin
        with np.load(filename) as src:b={k:src[k] for k in src}
        if stage:
            with np.load(root/'horizon/residual'/split/house/filename.name) as add:
                assert np.array_equal(add['paths'],b['paths']) and np.array_equal(add['goals'],b['goals'])
                b['horizon_jepa']=add['residual'];b['horizon_direct']=add['residual_direct']
        if stage=='layout':
            with np.load(root/'layout_update'/split/house/filename.name) as add:
                assert np.array_equal(add['paths'],b['paths']) and np.array_equal(add['goals'],b['goals'])
                b['adjust_jepa']=add['adjust_jepa']
        if stage=='transition':
            with np.load(root/'transition'/split/house/filename.name) as add:
                assert np.array_equal(add['paths'],b['paths']) and np.array_equal(add['goals'],b['goals'])
                b['transition']=add['env_pred_jepa']
        manifest=json.loads((raw/'episode.json').read_text())
        choices=[]
        for q,target in enumerate(manifest['targets']):
            word=target['word'];assert word==b['goals'][q]
            # Included as a separate absence control; no policy receives house_count or quota.
            agent=Junctions(b,q,old,project,cue)
            for seed in range(len(SEEDS)):
                for name,(method,config) in configs.items():
                    steps,scout=agent.choices(seed,method,config,'horizon' in name,'layout' in name,'transition' in name)
                    choices.append((target,seed,name,steps,scout))
        truth=json.loads((raw/'truth.json').read_text())
        for target,seed,name,steps,scout in choices:
            word=target['word'];ev=assess(truth,steps,word,target['quota'])
            logs.append(dict(house=house,origin=origin,word=word,stratum=target['stratum'],
                             quota=target['quota'],count=target['house_count'],seed=SEEDS[seed],method=name,
                             path=steps,scout_changed=scout,**ev))
    n=sum(1 for _ in (root/'simulator/trees'/split).glob('*/*/episode.json'))
    assert n==len({(r['house'],r['origin']) for r in logs})
    dst=online/(split+('_'+stage if stage else '')+'_decisions.jsonl'); temp=dst.with_suffix('.partial')
    with temp.open('w') as f:
        for row in logs:f.write(json.dumps(row)+'\n')
    temp.replace(dst)
    print(split,n,len(logs),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('project',type=Path)
    ap.add_argument('split',choices=('calibration','fresh'))
    ap.add_argument('--stage',default='',choices=('','horizon','layout','transition'))
    a=ap.parse_args();main(a.project,a.split,a.stage)
