"""Compare the frozen models when every available route costs exactly the same."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from train_prefix import distribution, frozen_options, write_new


def main(project):
    out=project/"jepa_variable_v16/model_training"
    old=project/"jepa_evidence_v15/train"
    sys.path[:0]=[str(old),str(project/"jepa_hidden_v11/models")]
    from train import load
    from evidence_head import EvidenceNet
    torch.set_num_threads(2)
    keys, data=load(project,"fresh2",False)
    dense=[]
    for _,origin,word in keys:
        with np.load(out/"private_fresh"/(origin+".npz")) as z:
            dense.append(z["visible"][z["words"].tolist().index(word)])
    dense=np.stack(dense)
    freeze=json.loads((out/"freeze.json").read_text())
    cal=json.loads((project/"jepa_variable_v16/model/calibration.json").read_text())
    x,text=(torch.from_numpy(data[k]).float() for k in ("x","text"))
    result={}
    for name,path in (("jepa",out/"jepa.pt"),("direct",out/"direct.pt"),
                      ("production_evidence_rate",old/"frozen.pt")):
        if name in freeze["choices"]:
            assert hashlib.sha256(path.read_bytes()).hexdigest()==freeze["choices"][name]["sha256"]
        net=EvidenceNet(text.shape[-1])
        net.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
        net.eval()
        with torch.inference_mode():
            z=torch.cat([net(xx,tt)[1] for xx,tt in zip(x.split(96),text.split(96))])
        table={}
        for n in (2,3):
            _, ids=frozen_options(keys,n)
            rows={}
            for j in (1,2,3,4):
                scores=distribution(z,torch.full(z.shape[:2],j))[1]
                if name=="production_evidence_rate":
                    scaled=cal["scale"]*z[:,:,:j]+cal["offset"]
                    scores=-torch.expm1(F.logsigmoid(-scaled).mean(-1))
                selected=ids.gather(1,scores.gather(1,ids).argmax(1,keepdim=True)).numpy().ravel()
                hit=dense[:,:,:6*j].any(-1)
                choices=hit[np.arange(len(hit))[:,None],ids.numpy()]
                rows[6*j]=dict(selected=int(hit[np.arange(len(hit)),selected].sum()),
                               random_expected=float(choices.sum()/n), oracle=int(choices.any(1).sum()))
            table[n]=rows
        result[name]=table
    write_new(out/"equal_cost.json",dict(queries=len(keys),unit="queries with a physical discovery at any step",results=result))
    for k,v in result.items():
        print(k,[[w["selected"] for w in v[n].values()] for n in (2,3)])


if __name__=="__main__":
    main(Path(sys.argv[1]))
