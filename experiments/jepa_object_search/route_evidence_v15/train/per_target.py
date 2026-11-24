"""Report fixed-checkpoint first-choice support for the two MOST pilot categories."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from evidence_head import EvidenceNet
from train import load, macro, write


def main(project):
    folder=project/"jepa_evidence_v15/train"
    selection=json.loads((folder/"selection.json").read_text())
    torch.set_num_threads(2)
    nets={}
    hashes={}
    for name, file in (("jepa","frozen.pt"),("direct","direct.pt")):
        hashes[name]=hashlib.sha256((folder/file).read_bytes()).hexdigest()
        net=EvidenceNet(selection["dim"]).eval()
        net.load_state_dict(torch.load(folder/file,map_location="cpu",weights_only=True))
        nets[name]=net
    assert hashes["jepa"]==selection["sha256"]
    words=["clock.n.01","wall_art.n.01"]
    summary={}
    for split in ("train","extra_train","calibration","test","fresh2"):
        keys,d=load(project,split,False)
        out={}
        for word in words:
            ix=[i for i,k in enumerate(keys) if k[2]==word]
            kk=[keys[i] for i in ix]
            y=d["y"][ix]>0
            rec=dict(queries=len(ix),houses=len({k[0] for k in kk}),positive_routes=int(y.sum()),
                     total_routes=int(y.size))
            if ix and split in ("test","fresh2"):
                selected={}
                for name,n in nets.items():
                    with torch.inference_mode():
                        prob=n(torch.from_numpy(d["x"][ix]),torch.from_numpy(d["text"][ix]))[0]
                    # Rank using actor inputs only; attach held-out outcomes below.
                    sel=prob.argmax(-1).numpy()
                    hit=y[np.arange(len(ix)),sel]
                    selected[name]=sel
                    rec[name]=dict(first_choice_hits=int(hit.sum()),micro=float(hit.mean()),
                                   house_macro=float(np.mean(list(macro(kk,hit).values()))))
                rec["different_choice"]=int(np.sum(selected["jepa"]!=selected["direct"]))
                rec["uniform_chance_micro"]=float(y.mean())
            out[word]=rec
        summary[split]=out
    write(folder/"per_target.json",dict(checkpoints_sha256=hashes,categories=words,splits=summary))
    print(json.dumps(summary))


if __name__=="__main__":
    main(Path(sys.argv[1]))
