"""Frozen inference for candidate camera configurations. No labels, map edits or route actions."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from train_joint import JointPredictor


class CameraPredictor:
    def __init__(self,checkpoint,device="cpu"):
        self.device=device
        saved=torch.load(checkpoint,weights_only=True,map_location=device)
        self.net=JointPredictor(saved["dims"]).to(device)
        self.net.load_state_dict(saved["weights"],strict=True)
        self.net.eval()
        self.variant=saved["variant"]
        self.stats=saved["stats"]["pose"].cpu().numpy()

    @torch.inference_mode()
    def __call__(self,context,action,pose):
        context=np.asarray(context,dtype=np.float32)
        action=np.asarray(action,dtype=np.int64)
        pose=np.asarray(pose,dtype=np.float32)
        if context.ndim!=2 or context.shape[1]!=2048 or pose.shape!=(len(context),len(self.stats[0])):
            raise ValueError("expected a 1024-D source plus 1024-D observed cue and matching camera feature rows")
        if action.ndim!=2 or len(action)!=len(context) or action.shape[1]<1 or not np.isin(action,[0,1,2,3]).all():
            raise ValueError("expected one primitive plan per row; ids 1/2/3, trailing padding 0")
        if ((action[:,:-1]==0)&(action[:,1:]!=0)).any(): raise ValueError("nontrailing padding")
        if not np.isfinite(context).all() or not np.isfinite(pose).all(): raise ValueError("nonfinite features")
        pose=np.clip((pose-self.stats[0])/self.stats[1],-8,8)
        t=lambda a:torch.as_tensor(a,device=self.device)
        out=self.net.predict("pose",t(context),t(action),t(pose),self.variant)
        names=("predicted_future","p_reach","p_visible_if_reach","p_joint","p_optical")
        return {name:v.cpu().numpy() for name,v in zip(names,out)}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint",type=Path)
    p.add_argument("observations",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    with np.load(a.observations,allow_pickle=False) as z:
        out=CameraPredictor(a.checkpoint)(z["context"],z["action"],z["pose"])
    np.savez_compressed(a.output,**out)
    print(json.dumps({"candidates":len(out["p_joint"])}))
