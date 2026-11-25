"""Evaluate a new supplementary manifest using the already frozen weights and thresholds."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from reid_model import OBJECT_DIM, ViewHead
from train import describe, extract, delta, make_pairs, predict


def main(a):
    rows=[json.loads(line) for line in a.manifest.read_text().splitlines()]
    a.out.mkdir(parents=True,exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    feats=extract(a.project,a.manifest,rows,a.out,device).to(device)
    results={}
    thresholds=json.loads((a.model/"metrics.json").read_text())["thresholds"]
    for split in a.splits:
        p=make_pairs(rows,split)
        ang=torch.tensor([delta(rows,i,j) for i,j,_ in p],dtype=torch.float32)
        values={}
        results[split]={}
        for name,t in thresholds.items():
            if name in ("direct","jepa","jepa_object"):
                ck=torch.load(a.model/(name+".pt"),map_location="cpu",weights_only=True)
                head=ViewHead().to(device)
                head.load_state_dict(ck["state_dict"])
                inp=feats
                if name=="jepa_object":
                    inp=feats.clone()
                    inp[:,OBJECT_DIM:]=0.
                    inp=F.normalize(inp,dim=-1)
                s=predict(head.eval(),inp,p,ang,device)
            else:
                dims={"frozen":slice(None),"object":slice(OBJECT_DIM),"context":slice(OBJECT_DIM,None)}[name]
                s=torch.cat([F.cosine_similarity(feats[q[:,0],dims],feats[q[:,1],dims]).cpu() for q in np.array_split(p,max(1,int(np.ceil(len(p)/256))))]).numpy()
            results[split][name]=describe(p[:,2],s,t)
            values["s_"+name]=s
        np.savez_compressed(a.out/("pairs_"+split+".npz"),i=p[:,0],j=p[:,1],y=p[:,2],**values)
    (a.out/"metrics.json").write_text(json.dumps(dict(thresholds=thresholds,metrics=results),indent=2)+"\n")
    print(results)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("project","manifest","model","out"):
        p.add_argument("--"+key,required=True,type=Path)
    p.add_argument("--splits",nargs="+",default=["test"])
    main(p.parse_args())
