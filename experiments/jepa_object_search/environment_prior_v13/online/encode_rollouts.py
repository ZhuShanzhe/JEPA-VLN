"""Cache observations separately from verdicts. Route scorers consume only departure views."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F


def done_write(dst, **fields):
    dst.parent.mkdir(parents=True,exist_ok=True)
    tmp=dst.with_suffix('.pending.npz')
    np.savez_compressed(tmp,**fields)
    os.replace(tmp,dst)


def encode(encoder, manifest, dst):
    info=json.loads(manifest.read_text())
    goals=[t['word'] for t in info['targets']]
    vocab=encoder.words(goals)
    text=F.normalize(vocab@encoder.proj,dim=-1).cpu().numpy()
    paths=[]; rgb=[]; actions=[]; bumps=[]
    for file in sorted(manifest.parent.glob('node_*.npz')):
        paths.append(file.stem[5:])
        with np.load(file) as z:
            frames=np.concatenate([z['ctx'],z['walk_rgb'].reshape(-1,*z['ctx'].shape[1:])])
            rgb.append(frames);actions.append(z['plans']);bumps.append(z['collided'])
    assert len(paths)==40 and paths[0]=='r'
    regions=[];cues=[]
    for i in range(0,len(paths),4):
        imgs=np.concatenate(rgb[i:i+4])
        feat,cue=encoder.encode(imgs,vocab)
        regions.extend(feat.reshape(-1,15,*feat.shape[1:]))
        cues.extend(cue.reshape(-1,15,len(goals)))
        if len(encoder.cache)>620:encoder.cache.clear()
    done_write(dst,paths=paths,goals=goals,text=text,features=regions,cues=cues,plans=actions,blocked=bumps)


def score(models,src,dst):
    with np.load(src) as z:
        goals=z['goals'].tolist(); paths=z['paths'].tolist()
        features=z['features'];plans=z['plans'];texts=z['text']
        cues=z['cues'];blocked=z['blocked']
    first=models[0]
    with torch.inference_mode():
        ctx=first.tensor(features.reshape(-1,*features.shape[2:]))
        sensed=first.reader(ctx).softmax(-1).cpu().numpy().reshape(len(paths),15,-1)
    arrays={}
    for m in models:
        local={}
        for i in range(len(paths)):
            per={}
            for q,word in enumerate(goals):
                out=m.cached(features[i,:3],texts[q],plans[i],word)
                for key in ('v11','direct','jepa','env_pred_direct','env_pred_jepa','env_pred_copy','prior'):
                    per.setdefault(key,[]).append(out[key])
            for key,val in per.items():local.setdefault(key,[]).append(val)
        for key,val in local.items():arrays.setdefault(key,[]).append(val)
    done_write(dst,paths=paths,goals=goals,sensed=sensed,cue=cues,plans=plans,blocked=blocked,**arrays)


def main(p, split, job):
    torch.set_num_threads(2)
    exp=p/'jepa_env_v13';tree=exp/'simulator/trees'/split
    cohort=json.loads((exp/'simulator/cohort.json').read_text())
    houses=cohort[split]
    if job=='encode':
        sys.path.insert(0,str(p/'jepa_hidden_v11/models'))
        from encode import Encoder
        encoder=Encoder(p)
    else:
        sys.path.insert(0,str(exp/'models'))
        from env_routes import EnvironmentRoutes
        models=[EnvironmentRoutes(p,seed) for seed in (11,29,47)]
    finished=set()
    while True:
        for manifest in sorted(tree.glob('*/*/episode.json')):
            house,origin=manifest.parent.parent.name,manifest.parent.name
            key=(house,origin)
            if key in finished:continue
            bank=exp/'online/visual'/split/house/(origin+'.npz')
            output=bank if job=='encode' else exp/'online/scored'/split/house/(origin+'.npz')
            if not output.exists():
                if job=='encode':encode(encoder,manifest,output)
                elif bank.exists():score(models,bank,output)
                else:continue
            finished.add(key)
        receipts=list(tree.glob('*/receipt.json'))
        if len(receipts)==len(houses) and len(finished)==len(list(tree.glob('*/*/episode.json'))):break
        time.sleep(4)
    print(job,split,'origins',len(finished),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('project',type=Path)
    p.add_argument('split',choices=['calibration','fresh']);p.add_argument('job',choices=['encode','score'])
    a=p.parse_args();main(a.project,a.split,a.job)
