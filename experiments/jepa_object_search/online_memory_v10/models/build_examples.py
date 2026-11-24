"""Build blind HSSD candidate features, then load physically measured route opportunity labels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

D=Path(__file__).resolve().parent
ROOT=D.parents[1]
BASE=ROOT/'jepa_five_20261123'
V8=BASE/'report_v8'
sys.path[:0]=[str(D),str(V8),str(BASE/'report_v7'),str(BASE/'clip_v5'),str(ROOT/'mapping_runtime/pilots/jepa_round2_five_20261121')]
from cache_observed import assets
from fit_v8 import restore_heads
from route_heads import load_heads
from grounded_route import evidence_fields,jepa_fields
from report_head import forecast_inputs
from v10_live import RouteMemory,DEFAULT
from torch.nn import functional as F

@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    enc,words=assets();ground,nets,words2=restore_heads(V8/'v8.pt');assert words==words2
    route=load_heads(BASE/'clip_v5/head_v5.pt',BASE/'sparse_v6/sparse_head.pt')
    anchors=enc.anchors;src=ROOT/'jepa_separation_v8_20261123/primitives'
    cache=D.parent/'features';cache.mkdir(exist_ok=True);init=D.parent/'initial.json'
    if not init.exists():init.write_text(json.dumps(dict(weights=DEFAULT)))
    for split in ['train','calibration','holdout']:
        rows=json.loads((src/(split+'.index.json')).read_text())
        for j,r in enumerate(rows):
            uid=hashlib.sha256(r['sid'].encode()).hexdigest();out=cache/(uid+'.npz')
            if out.exists():continue
            with np.load(D.parent/'prefixes'/(uid+'.npz')) as pre:
                rgbs=pre['rgb'];vec,ang,st,coll=pre['xy'],pre['heading'],pre['steps'],pre['collided']
                if int(st[-1])!=int(pre['frame_steps'][-1]):raise ValueError('missing current observation')
                frame_ids=pre['frame_steps'];translated=bool(pre['translated'])
            # Prefix can be real translation or a full spin. Historical visual features
            # are from that chronological prefix. No labels, poses or future views enter rank.
            local=enc.local(rgbs)
            code,anc,sim=enc.compressed(local,anchors)
            v,n=ground(code.half().float(),anc.half().float(),sim.half().float(),anchors)
            current_ctx=F.normalize(torch.tensor(enc.encoder.images(rgbs[-1:])).float(),dim=-1)
            cues=v.amax(1).numpy();score_at=dict(zip(frame_ids,cues))
            with np.load(r['observed']) as z:acts=z['actions']
            memories=[RouteMemory(init) for _ in words]
            for i,step in enumerate(st):
                cue=score_at.get(i)
                for q,m in enumerate(memories):
                    m.observe(vec[i],int(step),ang[i],None if cue is None else cue[q],coll[i])
            features=[]
            for h in (4,12,24):
                plans=acts[:,:h];a=torch.from_numpy(plans).long();hd=torch.zeros(1)
                p,near,_=forecast_inputs(route,'sparse',current_ctx,hd,a)
                dense,geom=evidence_fields(v[-1:],n[-1:],hd,a)
                latent=jepa_fields(p,near,anchors,anchors)
                frozen=[nets[k](dense,latent,geom)[0].sigmoid().numpy() for k in ('joint','observed','geometry')]
                mem=np.stack([m.features(plans,vec[-1],ang[-1],int(st[-1])) for m in memories],axis=1)
                features.append(np.concatenate([np.stack(frozen,-1),mem],-1))
            # Only evaluator / pairwise fitting sees these array fields.
            with np.load(r['labels']) as gt:
                ix=[list(gt['queries']).index(w) for w in words]
                gains=gt['gain'].transpose(1,0,2)[:,:,ix]
                present=gt['query_present'][ix]
            tmp=out.with_suffix('.pending.npz')
            np.savez_compressed(tmp,f=np.array(features),y=gains,present=present,actions=acts,
                translated=translated,steps=len(st),keyframes=len(frame_ids),words=words,
                observations_sha=hashlib.sha256(rgbs.tobytes()).hexdigest())
            tmp.replace(out)
            if j%10==0:print(split,j+1,'/',len(rows),flush=True)
        print('READY',split,len(rows),flush=True)
if __name__=='__main__':main()
