"""Fit only on development training houses and select search heads on calibration houses."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.nn import functional as F

from env_routes import Reader,RouteHead,dependencies,digest,prior_vector

SEEDS=(11,29,47)
STEPS=(180,360,540)
MIXES=(.15,.35,.6,1.)


def records(project,split,old,raw):
    exp=project/'jepa_env_v13'; data=[]
    cache=(project/('jepa_route_v12' if split=='extra_train' else 'jepa_hidden_v11')/'models/cache/hssd'/split)
    mem=old.remem(project)
    for p in sorted((cache/'actor').glob('*.npz')):
        with np.load(p) as z:
            ctx=z['ctx'].astype('float32');plans=z['actions'];h=str(z['house']);o=str(z['origin'])
            words=z['words'].tolist();text=z['text'].astype('float32');cue=z['cue']
        with np.load(exp/'data/labels'/split/(o+'.npz')) as z:
            room=z['room_idx'].astype('int64');start=int(z['start_room'])
            view=np.array(z['view_room'],np.float32)
            if np.any(~np.isfinite(view)) or np.any(view<0):raise ValueError('invalid visible-room evidence')
            total=view.sum(-1,keepdims=True)
            empty=(total[...,0]<=1e-7)&(room>=0)
            view=view/np.maximum(total,1e-7)
            view[empty]=np.eye(len(raw['room_names']),dtype='float32')[room[empty]]
            start_view=np.array(z['start_view_room'],np.float32)
            svsum=start_view.sum(-1,keepdims=True)
            start_view=start_view/np.maximum(svsum,1e-7)
            if start>=0:start_view[svsum[:,0]<1e-7]=np.eye(len(raw['room_names']),dtype='float32')[start]
        with np.load(cache/'teacher'/p.name) as z:
            future=z['future'].astype('float32')
        with np.load(cache/'truth'/p.name) as z:
            gains=z['y'].T
        x=old.route_inputs(ctx,plans)
        for i,w in enumerate(words):
            old.memory_start(mem,cue[:,i])
            data.append(dict(x=x,ctx=ctx,future=future,rooms=room,view=view,start=start,start_view=start_view,target=text[i],
                      prior=prior_vector(raw,w),y=gains[i],h=h,o=o,word=w,bonus=old.bonus(mem,plans,2)))
    return data


def pack(rows,device):
    return {k:torch.as_tensor(np.stack([r[k] for r in rows]),device=device,
                  dtype=torch.long if k in ('rooms','start') else torch.float32)
            for k in ('x','future','rooms','view','target','prior','y','ctx','start')}


def sampler(rows):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['h']].append(i)
    keys=sorted(groups)
    def draw(rng,n):
        return [rng.choice(groups[keys[k]]) for k in rng.integers(len(keys),size=n)]
    return draw


def macro(rows,vals):
    origins=defaultdict(list)
    for r,v in zip(rows,vals):origins[r['h'],r['o']].append(v)
    houses=defaultdict(list)
    for (h,_),a in origins.items():houses[h].append(np.mean(a))
    return {h:float(np.mean(a)) for h,a in houses.items()}


def fit_reader(rows,cal,k,device,d):
    def views(rset):
        f=[];y=[];h=[];seen=set()
        for r in rset:
            if r['o'] in seen:continue
            seen.add(r['o'])
            feat=np.concatenate([r['ctx'],r['future'].reshape(12,3,-1)])
            labs=np.concatenate([r['start_view'],r['view'].reshape(12,-1)])
            valid=labs.sum(-1)>0
            f.extend(feat[valid]);y.extend(labs[valid]);h.extend([r['h']]*sum(valid))
        return torch.tensor(np.stack(f),device=device),torch.tensor(np.stack(y),device=device),h
    tf,ty,hs=views(rows);vf,vy,hv=views(cal)
    torch.manual_seed(817);reader=Reader(tf.shape[-1],k).to(device)
    opt=torch.optim.AdamW(reader.parameters(),lr=.00075)
    draw=sampler([{'h':h} for h in hs]);rng=np.random.default_rng(913)
    for step in range(360):
        ids=draw(rng,128);v=reader(tf[ids]);loss=F.cross_entropy(v,ty[ids])
        opt.zero_grad();loss.backward();opt.step()
    reader.eval()
    with torch.no_grad():
        preds=reader(vf).argmax(-1);correct=(preds==vy.argmax(-1)).cpu().numpy()
    path=d/'weights/reader.pt';torch.save(reader.state_dict(),path)
    hh=np.array(hv)
    majority=int(ty.sum(0).argmax())
    return dict(hashes={'reader':digest(path)},train_views=len(tf),cal_views=len(vf),train_majority=majority,
        cal_view_accuracy=float(correct.mean()),cal_house_accuracy=float(np.mean([correct[hh==h].mean() for h in set(hv)])),
        majority_cal_house_accuracy=float(np.mean([(vy.argmax(-1).cpu().numpy()[hh==h]==majority).mean() for h in set(hv)])))


def score_choices(rows,y,scores):
    sel=scores.argmax(-1);correct=y[np.arange(len(rows)),sel]>0
    homes=macro(rows,correct)
    return float(np.mean(list(homes.values())))


def main(project):
    d=project/'jepa_env_v13/models';(d/'weights').mkdir(exist_ok=True)
    raw=json.loads((project/'jepa_env_v13/data/prior.json').read_text());k=len(raw['room_names'])
    shutil.copyfile(project/'jepa_env_v13/data/prior.json',d/'prior.json')
    old,load_prior=dependencies(project)
    tr=records(project,'train',old,raw)+records(project,'extra_train',old,raw);cal=records(project,'calibration',old,raw)
    assert {r['h'] for r in tr}.isdisjoint({r['h'] for r in cal})
    device='cuda';torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    trdata=pack(tr,device);cv=pack(cal,device);draw=sampler(tr)
    yy=cv['y'].cpu().numpy();bonus=np.stack([r['bonus'] for r in cal])
    selected=dict(room_names=raw['room_names'],train_houses=len({r['h'] for r in tr}),cal_houses=len({r['h'] for r in cal}),
                  train_queries=len(tr),cal_queries=len(cal),seeds=list(SEEDS),prior_sha256=digest(d/'prior.json'))
    selected['reader']=fit_reader(tr,cal,k,device,d);table=defaultdict(lambda:defaultdict(list))
    for seed in SEEDS:
        baseline=load_prior(old,project,seed,'jepa',device)
        with torch.no_grad():prior=baseline(cv['x'],cv['target'])[0].sigmoid().cpu().numpy()
        selected.setdefault('baseline_cal',[]).append(score_choices(cal,yy,prior+bonus))
        for arm in ('jepa','direct'):
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=RouteHead(trdata['target'].shape[-1],k).to(device)
            optim=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.015)
            for step in range(1,max(STEPS)+1):
                ids=draw(rng,48)
                x,t,p,y,classes=(trdata[key][ids] for key in ('x','target','prior','y','view'))
                logits,room,latent=model(x,t,p)
                valid=classes.sum(-1)>0;ce=(F.cross_entropy(room[valid],classes[valid]) if valid.any() else room.sum()*0.)
                target=(y>0).float();rankmask=y[:,:,None]>y[:,None,:]
                rank=(F.softplus(logits[:,None,:]-logits[:,:,None])*rankmask).sum()/rankmask.sum().clamp_min(1)
                loss=F.binary_cross_entropy_with_logits(logits,target)+rank+.6*ce
                if arm=='jepa':
                    u=trdata['future'][ids,:,:,0,:]
                    da=F.normalize(latent-latent.mean(1,keepdim=True),dim=-1)
                    db=F.normalize(u-u.mean(1,keepdim=True),dim=-1)
                    loss=loss+.8*(1-(u*latent).sum(-1).mean())+.2*(1-(da*db).sum(-1).mean())
                optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optim.step()
                if step in STEPS:
                    torch.save(model.state_dict(),d/'weights'/f'{arm}_{seed}_{step}.pt')
                    with torch.no_grad():
                        prediction,rm,_=model(cv['x'],cv['target'],cv['prior'])
                        pred=prediction.sigmoid().cpu().numpy();classes=cv['view'].cpu().numpy()
                        known=classes.sum(-1)>0;ok=(rm.argmax(-1).cpu().numpy()==classes.argmax(-1))
                        # Do not weight identical route-room labels once per query.
                        unique=np.unique([r['o'] for r in cal],return_index=True)[1]
                        room_accuracy=float(ok[unique][known[unique]].mean())
                    for mix in MIXES:
                        table[arm][step,mix].append((score_choices(cal,yy,(1-mix)*prior+mix*pred+bonus),room_accuracy))
            print(arm,seed,flush=True)
    for arm in table:
        grid={key:np.mean(vals,axis=0) for key,vals in table[arm].items()}
        step,mix=max(grid,key=lambda k:(grid[k][0],-k[1],-k[0]))
        codes=[f'{arm}_{s}_{step}' for s in SEEDS]
        matched=grid[360,.35]
        selected[arm]=dict(step=step,mix=mix,cal_first=float(grid[step,mix][0]),
            cal_room_route_tap_accuracy=float(grid[step,mix][1]),cal_matched_step360_mix035=matched.tolist(),
            hashes={s:digest(d/'weights'/(s+'.pt')) for s in codes})
    (d/'selection.json').write_text(json.dumps(selected,indent=2))
    print('frozen',selected['reader'],[(a, selected[a]['step'],selected[a]['mix']) for a in table],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('project',type=Path);main(ap.parse_args().project)
