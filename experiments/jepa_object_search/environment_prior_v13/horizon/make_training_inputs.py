"""Generate actor-only route forecasts for counterfactually relabeled training trees."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F


@torch.inference_mode()
def make(models,source,dest,prior_vector):
    one=models[0]; infos=json.loads(source.read_text())['targets']; goals=[v['word'] for v in infos]
    t=one.encoder.words(goals);texts=F.normalize(t@one.encoder.proj,dim=-1);q=len(goals)
    prior=one.tensor(np.stack([prior_vector(one.raw,g) for g in goals]))
    paths=[];images=[];plans=[]
    for p in sorted(source.parent.glob('node_*.npz')):
        paths.append(p.stem[5:])
        with np.load(p) as z:images.append(z['ctx']);plans.append(z['plans'])
    assert len(plans)==40
    ctx,obs=one.encoder.encode(np.concatenate(images),t)
    ctx=ctx.reshape(40,3,3,-1); obs=obs.reshape(40,3,q)
    if len(one.encoder.cache)>600:one.encoder.cache.clear()
    x=one.tensor(np.stack([one.old.route_inputs(c.astype('float32'),p) for c,p in zip(ctx,plans)]))
    x=x[:,None].expand(-1,q,-1,-1,-1).flatten(0,1)
    txt=texts[None].expand(40,-1,-1).flatten(0,1);pr=prior[None].expand(40,-1,-1).flatten(0,1)
    sensed=one.reader(one.tensor(ctx)).softmax(-1).cpu().numpy()
    fields={}
    for m in models:
        v11=m.base(x,txt)[0].sigmoid()
        for arm in ('direct','jepa'):
            score,env,_=m.heads[arm](x,txt,pr)
            mix=m.config[arm]['mix'];v=(1-mix)*v11+mix*score.sigmoid()
            fields.setdefault(arm,[]).append(v.reshape(40,q,3).cpu().numpy())
            fields.setdefault('env_pred_'+arm,[]).append(env.softmax(-1).mean(2).reshape(40,q,3,-1).cpu().numpy())
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,paths=paths,goals=goals,sensed=sensed,cue=obs,plans=plans,
            prior=np.broadcast_to(prior.cpu().numpy(),(3,40,q,prior.shape[-1])),**fields)
    tmp.replace(dest)


def main(root):
    sys.path.insert(0,str(root/'models'))
    from env_routes import EnvironmentRoutes,prior_vector
    models=[EnvironmentRoutes(root.parent,seed) for seed in (11,29,47)]
    done=set()
    while True:
        for p in sorted((root/'simulator/trees/train').glob('*/*/training_targets.json')):
            h,o=p.parent.parent.name,p.parent.name
            if (h,o) in done:continue
            dest=root/'horizon/banks'/h/(o+'.npz')
            if not dest.exists():make(models,p,dest,prior_vector)
            done.add((h,o))
        if len(list((root/'simulator/trees/train').glob('*/receipt.json')))==12 and len(done)==48:break
        time.sleep(2)
    print('encoded',len(done),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('experiment',type=Path);main(ap.parse_args().experiment)
