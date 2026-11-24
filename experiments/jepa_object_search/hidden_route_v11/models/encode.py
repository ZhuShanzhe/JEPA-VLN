"""Freeze RN50 observations and future teacher targets in separate caches."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


class Encoder:
    def __init__(self, root):
        base = root / 'jepa_five_20261123'
        sys.path[:0] = [str(base/'clip_v5'), str(base/'report_v8')]
        from clip_views import Clip
        from grounded_route import LocalGrounder
        self.clip = Clip()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = self.clip.net.to(self.device).eval()
        v5 = torch.load(base/'clip_v5/head_v5.pt', map_location=self.device, weights_only=True)
        ck = torch.load(base/'report_v8/v8.pt', map_location=self.device, weights_only=True)['grounding']
        self.ground = LocalGrounder(ck['anchors']).to(self.device).eval()
        self.ground.load_state_dict(ck)
        self.center, self.basis = v5['center'], v5['basis']
        # Shared fixed subspace. It never uses this study's future or test images.
        gen = torch.Generator().manual_seed(719)
        rand = torch.randn((1024, 32),generator=gen).to(self.device)
        self.proj = torch.linalg.qr(torch.cat([self.basis,ck['anchors'].T,rand],dim=1)).Q
        self.texts = {}
        self.cache = {}

    @torch.inference_mode()
    def words(self, words):
        import re
        for w in words:
            if w not in self.texts:
                name = re.sub(r'\.n\.\d+$','',w).replace('_',' ')
                tok = self.clip.tokenize(['a photo of a '+name+'.']).to(self.device)
                self.texts[w] = F.normalize(self.net.encode_text(tok).float(),dim=-1)[0]
        return torch.stack([self.texts[w] for w in words])

    @torch.inference_mode()
    def encode(self, images, text):
        vis = self.net.visual
        pool = vis.attnpool
        keys = [hashlib.sha256(x.tobytes()).digest() for x in images]
        missing = {key:x for key,x in zip(keys,images) if key not in self.cache}
        ids = list(missing)
        for j in range(0,len(ids),24):
            kk = ids[j:j+24]; tapped=[]
            hook = pool.register_forward_pre_hook(lambda _,args:tapped.append(args[0]))
            try:
                imgs = torch.stack([self.clip.prep(Image.fromarray(missing[k])) for k in kk]).to(self.device)
                whole = F.normalize(vis(imgs).float(),dim=-1)
            finally:
                hook.remove()
            grid = tapped[0].flatten(2).transpose(1,2)
            tokens = F.normalize(pool.c_proj(pool.v_proj(grid)).float(),dim=-1)
            sides = tokens.reshape(-1,7,7,1024)
            regions = torch.stack([whole,F.normalize(sides[:,:,:3].mean((1,2)),dim=-1),
                                   F.normalize(sides[:,:,4:].mean((1,2)),dim=-1)],dim=1)
            features = F.normalize(regions@self.proj,dim=-1).half().cpu().numpy()
            dense = tokens.half().cpu().numpy()
            for i,k in enumerate(kk):
                self.cache[k] = features[i],dense[i]
        regions = np.stack([self.cache[k][0] for k in keys])
        grid = torch.from_numpy(np.stack([self.cache[k][1] for k in keys])).to(self.device,dtype=torch.float32)
        codes = ((grid-self.center)@self.basis).half().float()
        a = (grid@self.ground.anchors.T).half().float()
        q = (grid@text.T).half().float()
        visible,_ = self.ground(codes,a,q,text)
        return regions,visible.amax(1).cpu().numpy()


def encode_one(enc, path, dest, source, split):
    with np.load(path,allow_pickle=False) as z:
        words=np.atleast_1d(z['words']).astype(str).tolist()
        ang=(np.asarray(z['headings'],float)+180)%360-180
        index=[int(np.argmin(abs(ang-v))) for v in [-30,0,30]]
        assert np.allclose(ang[index],[-30,0,30])
        plans=np.array(z['actions'],dtype=np.int64)
        assert plans.shape[0]>=2 and np.isin(plans,[0,1,2]).all()
        text=enc.words(words)
        ctx,cs=enc.encode(z['ctx'][index],text)
        root=dict(house=str(z['house'].item()),origin=str(z['origin'].item()),source=source,
                  split=split,actions=plans,words=words,ctx=ctx,cue=cs,
                  text=F.normalize(text@enc.proj,dim=-1).cpu().numpy())
        f=z['future']; c,k=f.shape[:2]
        assert c==len(plans) and k==4
        fu,fc=enc.encode(f.reshape(-1,*f.shape[2:]),text)
        y=np.asarray(z['y']);assert y.shape==(c,len(words))
        group=np.broadcast_to(np.array(z['group']).astype(str), (len(words),))
        more={k:z[k] for k in ('collided','instance_ids','hit_instance_ids','instance_counts','house_instances','room_instance_counts','sample_type','tags','manifest_index') if k in z and not z[k].dtype.hasobject}
    dest.parent.mkdir(exist_ok=True,parents=True)
    temp=dest.with_suffix('.pending.npz')
    np.savez_compressed(temp,**root)
    for kind,data in [('teacher',dict(future=fu.reshape(c,k,*fu.shape[1:]),future_cue=fc.reshape(c,k,-1),**more)),
                      ('truth',dict(y=y,group=group,**{k:v for k,v in more.items() if k not in ('collided',)}))]:
        other=dest.parent.parent/kind/dest.name; other.parent.mkdir(parents=True,exist_ok=True)
        part=other.with_suffix('.pending.npz');np.savez_compressed(part,**data);part.replace(other)
    temp.replace(dest)
    if len(enc.cache)>900:enc.cache.clear()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('project_root',type=Path);ap.add_argument('splits',nargs='+')
    a=ap.parse_args(); exp=a.project_root/'jepa_hidden_v11'
    torch.set_num_threads(2);enc=Encoder(a.project_root)
    for split in a.splits:
        for src in ('hssd','blender'):
            done=0
            for p in sorted((exp/src/split).glob('*/*.npz')):
                if p.name.endswith(('.pending.npz','.partial.npz')):continue
                key=hashlib.sha256(str(p.relative_to(exp)).encode()).hexdigest()
                out=exp/'models/cache'/src/split/'actor'/(key+'.npz')
                if out.is_file():continue
                encode_one(enc,p,out,src,split);done+=1
                if done%100==0:print(src,split,done,flush=True)
            print('encoded',src,split,done,flush=True)

if __name__=='__main__':main()
