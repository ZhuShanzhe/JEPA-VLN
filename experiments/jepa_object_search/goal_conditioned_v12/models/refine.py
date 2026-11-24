"""Calibrate a second foreground teacher and score locked equal-update heads."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from train_routes import (ARMS, MIX, SEEDS, STEPS, GoalRanker, digest, frozen_scores, load_prior,
                          load_rows, measure, pairs, previous, score_batches, train_once, write)


def criterion(metrics):
    return .7*metrics[0]+.3*metrics[1]


def refine(project):
    out=project/'jepa_route_v12'/'models'
    path=out/'deployment.json'
    if path.exists():
        raise FileExistsError(path)
    old=previous(project)
    tk,tr=load_rows(project,'train',True,True)
    ck,ca=load_rows(project,'calibration',True)
    goals={}
    targets=[]
    for _,origin,query in tk:
        if origin not in goals:
            with np.load(out.parent/'data'/'crop_targets'/'train'/f'{origin}.npz') as f:
                goals[origin]=(f['words'].tolist(),f['first_tap'],f['local_z'].astype('float32'))
        words,first,latent=goals[origin]
        idx=words.index(query)
        assert np.array_equal(first[:,idx],tr['t'][len(targets)].cpu().numpy())
        targets.append(latent[:,idx])
    tr['z']=torch.as_tensor(np.stack(targets),device='cuda')
    opposite=pairs(tk,tr['y'].cpu().numpy())
    y=ca['y'].cpu().numpy()
    cp=pairs(ck,y)
    mem=ca['memory'].cpu().numpy()
    destination=out/'crop_weights'
    destination.mkdir(exist_ok=True)
    results={}
    for seed in SEEDS:
        prior=score_batches(load_prior(old,project,seed,'jepa','cuda'),ca)
        model=train_once(tr,tk,opposite,seed,'v12_jepa',destination)
        for step in STEPS:
            model.load_state_dict(torch.load(destination/f'v12_jepa_{seed}_{step}.pt',weights_only=True))
            score=score_batches(model.eval(),ca)
            for alpha in MIX:
                j=np.argmax(mem+alpha*score+(1-alpha)*prior,axis=1)
                results.setdefault((step,alpha),[]).append(measure(ck,y,j,cp))
    avg={k:np.mean(v,axis=0).tolist() for k,v in results.items()}
    step,alpha=max(avg,key=lambda k:(criterion(avg[k]),avg[k][0],-k[0],-k[1]))
    settings=json.loads((out/'selection.json').read_text())['arms']
    base=settings['v12_jepa']
    ref=base['calibration'][f"{base['step']}/{base['head_weight']}"]
    choice='v12_jepa_crop' if criterion(avg[step,alpha])>criterion(ref) else 'v12_jepa'
    weights={}
    for seed in SEEDS:
        src=destination/f'v12_jepa_{seed}_{step}.pt'
        dest=out/'weights'/f'v12_jepa_crop_{seed}.pt'
        shutil.copyfile(src,dest)
        weights[str(seed)]=digest(dest)
    settings['v12_jepa_crop']=dict(step=step,head_weight=alpha,checkpoint_sha256=weights,
                                   calibration={f'{k[0]}/{k[1]}':v for k,v in avg.items()})
    common={}
    for arm in settings:
        folder=destination if arm=='v12_jepa_crop' else out/'weights'
        name='v12_jepa' if arm=='v12_jepa_crop' else arm
        common[arm]={str(s):digest(folder/f'{name}_{s}_200.pt') for s in SEEDS}
    write(path,dict(calibration_metric='0.7 first hit + 0.3 matched-minus-swapped goal margin',
                    selected=choice,mask_criterion=criterion(ref),crop_criterion=criterion(avg[step,alpha]),
                    training_complete_before_fresh_evaluation=True,arms=settings,matched_step=200,
                    matched_head_weight=.75,matched_sha256=common))
    print('deployment',choice,'mask/crop',criterion(ref),criterion(avg[step,alpha]),flush=True)


@torch.inference_mode()
def evaluate(project,split,matched):
    out=project/'jepa_route_v12'/'models'
    old=previous(project)
    lock=json.loads((out/'deployment.json').read_text())
    keys,tbl=load_rows(project,split)
    rosters=json.loads((project/'jepa_hidden_v11/hssd/cohort.json').read_text())
    blocked=set(rosters['train'])
    if split=='fresh':
        blocked.update(rosters['calibration']+rosters['test'])
    assert all(k[0] not in blocked for k in keys)
    target=out/(f'{split}_matched200_choices.jsonl' if matched else f'{split}_choices.jsonl')
    tmp=target.with_suffix('.pending')
    with tmp.open('w') as stream:
        for s in SEEDS:
            priors={'v11_'+a:load_prior(old,project,s,a,'cuda') for a in ('jepa','direct')}
            heads,mix={},{}
            for arm,settings in lock['arms'].items():
                if matched:
                    folder=out/'crop_weights' if arm=='v12_jepa_crop' else out/'weights'
                    name='v12_jepa' if arm=='v12_jepa_crop' else arm
                    weight=folder/f'{name}_{s}_200.pt'
                    expected=lock['matched_sha256'][arm][str(s)]
                    mix[arm]=lock['matched_head_weight']
                else:
                    weight=out/'weights'/f'{arm}_{s}.pt'
                    expected=settings['checkpoint_sha256'][str(s)]
                    mix[arm]=settings['head_weight']
                assert digest(weight)==expected
                model=GoalRanker(tbl['text'].shape[-1]).cuda()
                model.load_state_dict(torch.load(weight,weights_only=True))
                heads[arm]=model.eval()
            for start in range(0,len(keys),128):
                x=[tbl[k][start:start+128] for k in ('x','text','memory')]
                scores=frozen_scores(x,heads,priors,mix)
                mem=x[-1].cpu().tolist()
                for arm,raw in scores.items():
                    for j,row in enumerate(raw.cpu().numpy()):
                        h,o,q=keys[start+j]
                        stream.write(json.dumps(dict(split=split,arm=arm,seed=s,house=h,origin=o,
                            query=q,score=row.tolist(),initial_memory=mem[j],first=int(row.argmax())))+'\n')
    tmp.replace(target)
    print('locked choices',target.name,len(keys)*len(SEEDS)*5,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('project',type=Path)
    parser.add_argument('operation',choices=('refine','calibration','test','fresh'))
    parser.add_argument('--matched',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    if args.operation=='refine':
        refine(args.project)
    else:
        evaluate(args.project,args.operation,args.matched)
