"""Observed-only features for four-action physical navigation branches."""
import hashlib
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

PLANS=torch.tensor([[1,1,2,2],[2,2,1,1],[1,0,0,2],[2,0,0,1],[0,0,1,2],[0,0,2,1]])


@torch.inference_mode()
def route_fields(plans,crops,local_code,local_sim,v9,heads,words):
    from grounded_route import evidence_fields
    heading=torch.zeros(1)
    visible,near=v9(local_code,local_sim)
    geom=[];observed=[];future=[]
    for t in range(1,5):
        a=plans[:,:t]
        d,g=evidence_fields(visible,near,heading,a)
        d=d[:,-1];g=g[:,-1]
        # Retain nominal revealed view, potential near-target region and target bearing.
        observed.append(torch.stack([d[:,:,:4].amax(-1),d[:,:,4:8].amax(-1),d[:,:,8:12].amax(-1),
                                     d[:,:,12:16].amax(-1)],-1))
        # The reused sparse predictor accepts nominal motion and current observed crops only.
        predicted,nearest,_=heads[1](crops,heading,a)
        sim=predicted[:,-1]@words.T
        delta=(predicted[:,-1]-nearest[:,-1])@words.T
        # Preserve left and right crop conditioning rather than pooling opposite bearings.
        future.append(torch.cat([sim.transpose(1,2),delta.transpose(1,2)],-1))
        geom.append(g[:,[0,1,2,3,5]])
    gm=torch.cat(geom,-1)[:,None,:].expand(-1,len(words),-1)
    return dict(geo=gm.numpy(),observed=torch.cat(observed,-1).numpy(),jepa=torch.cat(future,-1).numpy())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ["v8-code","v9-code","v9","full","sparse","states","output"]:
        ap.add_argument("--"+name,required=True,type=Path)
    args=ap.parse_args()
    sys.path[:0]=[str(args.v9_code),str(args.v8_code)]
    from v9_core import Local
    from route_heads import load_heads
    torch.set_num_threads(2)
    models=load_heads(args.full,args.sparse)
    ck=torch.load(args.v9,weights_only=True,map_location="cpu")
    local=Local(len(ck["words"])).eval()
    local.load_state_dict(ck["local"])
    data=torch.load(args.states,weights_only=True,map_location="cpu")
    if data["words"]!=ck["words"]:raise ValueError("mismatched text features")
    for state in data["states"]:
        if set(state)!={"sid","split","crops","code","sim"}:raise ValueError("expected a causal observation cache")
        file=args.output/"features"/state["split"]/(hashlib.sha256(state["sid"].encode()).hexdigest()+".npz")
        file.parent.mkdir(parents=True,exist_ok=True)
        if file.exists():raise FileExistsError(file)
        f=route_fields(PLANS,state["crops"].float(),state["code"].float(),state["sim"].float(),local,models,data["anchors"])
        np.savez_compressed(file,words=data["words"],plans=PLANS.numpy(),**f)


if __name__=="__main__":
    main()
