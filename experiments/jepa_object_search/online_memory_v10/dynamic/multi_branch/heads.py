"""Fit future-embedding and direct-label priors; serve frozen heads over a local process pipe."""
import hashlib
import json
from multiprocessing.connection import Connection
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-9)


class Features:
    def __init__(self,project):
        sys.path.insert(0,str(project/"jepa_five_20261123/clip_v5"))
        from clip_views import Clip
        self.clip=Clip();self.cache={}

    @torch.inference_mode()
    def get(self,images,typ):
        keys=[(typ,hashlib.sha256(np.ascontiguousarray(im)).digest()) for im in images]
        missing={k:im for k,im in zip(keys,images) if k not in self.cache}
        for j in range(0,len(missing),12):
            batch=[]
            group=list(missing)[j:j+12]
            for k in group:
                im=missing[k]
                roi=im[80:256,118:248] if typ=="entrance" else im[100:256,62:194]
                batch.append(self.clip.prep(Image.fromarray(np.ascontiguousarray(roi))))
            z=F.normalize(self.clip.net.encode_image(torch.stack(batch)).float(),dim=-1).numpy()
            self.cache.update(zip(group,z))
        return np.array([self.cache[k] for k in keys])


def main(d,root,mode,fd=None):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    enc=Features(root);ck=d/"frozen.npz"
    if mode=="train":
        rows=[];detector={}
        for house in json.loads((d/"assets/protocol.json").read_bytes())["train_houses"]:
            f=d/"raw"/f"tri_{house:02}"
            for row in json.loads((f/"manifest.json").read_bytes()):
                starts=[np.array(Image.open(f/(k+".png")).convert("RGB")) for k in row["start"]]
                ends=[np.array(Image.open(f/(r["frame"]+".png")).convert("RGB")) for r in row["future"]]
                x=enc.get(starts,"entrance");y=enc.get(ends,"terminal")
                labels=np.eye(3)[[r["category"] for r in row["future"]]]
                rows.append((x,y,labels))
                for r,v in zip(row["future"],y):detector[r["frame"]]=(v,r["category"])
        x,y,cls=[np.concatenate(a).astype(np.float64) for a in zip(*rows)]
        cx=x.mean(0);cy=y.mean(0);src=unit(x-cx);future=unit(y-cy)
        # Both priors see identical entrance crops and target labels. Future features
        # supervise the latent head; direct labels supervise the matched control.
        K=src@src.T+np.eye(len(src))*1.
        latent=np.linalg.solve(K,future);direct=np.linalg.solve(K,cls-1./3)
        proto=unit(cls.T@future)
        detx=np.array([v[0] for v in detector.values()]);detcls=np.eye(3)[[v[1] for v in detector.values()]]
        cd=detx.mean(0);det=unit(detx-cd);da=np.linalg.solve(det@det.T+np.eye(len(det))*.75,detcls)
        sim=src@src.T
        scores=np.stack([(sim@latent)@proto.T,sim@direct])
        scale=np.maximum(1e-5,np.median(np.abs(scores-scores.mean(-1,keepdims=True)),axis=(1,2)))
        np.savez_compressed(ck,cx=cx,src=src,latent=latent,direct=direct,proto=proto,cd=cd,det=det,da=da,scale=scale)
        print("FROZEN",len(rows),len(det),hashlib.sha256(ck.read_bytes()).hexdigest(),flush=True)
    else:
        m=dict(np.load(ck));conn=Connection(int(fd))
        while True:
            try:msg=conn.recv()
            except EOFError:break
            if msg is None:break
            typ,imgs=msg;v=enc.get(imgs,typ)
            if typ=="terminal":
                out=unit(v-m["cd"])@m["det"].T@m["da"]
                conn.send(out.tolist())
            else:
                src=unit(v-m["cx"]);phi=src@m["src"].T
                z=np.stack([(phi@m["latent"])@m["proto"].T,phi@m["direct"]])
                z-=z.mean(axis=2,keepdims=True)
                score=.18*np.tanh(z/m["scale"][:,None,None])
                conn.send((score.tolist(),np.maximum(0,src@src.T).tolist()))


if __name__=="__main__":main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve(),*sys.argv[3:])
