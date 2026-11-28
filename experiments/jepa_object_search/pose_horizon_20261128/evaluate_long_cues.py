"""Test how the trained local visibility head extrapolates to physical long routes with an observed cue."""
import argparse
import json
from pathlib import Path
import numpy as np

from eval_pose import CameraPredictor
from train_joint import binary

HERE=Path(__file__).resolve().parent


def scores(reach, visible, probabilities):
    assert np.all(np.isin(reach,[0,1])) and np.all(np.isin(visible[reach==1],[0,1]))
    y=(reach==1)&(visible==1)
    return {"joint":binary(y,probabilities["p_joint"]),
            "reach":binary(reach,probabilities["p_reach"]),
            "visible_given_reached":binary(visible[reach==1],probabilities["p_visible_if_reach"][reach==1])}


def run(shards,output):
    import torch
    torch.set_num_threads(3)
    keys=("context","action","pose","reach","visible","house")
    blocks=[]
    for path in sorted(shards.glob("*.npz")):
        with np.load(path,allow_pickle=False) as z:
            if not (np.asarray(z["split"])=="test").all():continue
            blocks.append({k:np.array(z[k]) for k in keys})
    if not blocks:raise ValueError("missing held-out long-cue samples")
    data={k:np.concatenate([b[k] for b in blocks]) for k in keys}
    h=(data["action"]>0).sum(1)
    assert set(h)=={12,24,52,80,108}
    fixed=np.isclose(data["pose"][:,2],1.31) & np.isclose(data["pose"][:,6],0.)
    result=dict(houses=len(np.unique(data["house"])),rows=len(h),variants={})
    for variant in ("joint_long","planar_only","direct_outcome"):
        model=CameraPredictor(HERE/"private_train"/(variant+".pt"),"cuda" if torch.cuda.is_available() else "cpu")
        predictions={}
        for ix in np.array_split(np.arange(len(h)),max(1,(len(h)+127)//128)):
            p=model(data["context"][ix],data["action"][ix],data["pose"][ix])
            for key in ("p_joint","p_reach","p_visible_if_reach"):
                predictions.setdefault(key,[]).append(p[key])
        predictions={k:np.concatenate(v) for k,v in predictions.items()}
        groups={"all":np.ones_like(fixed),"fixed_camera":fixed}
        groups.update({str(v)+"_fixed":fixed&(h==v) for v in sorted(set(h))})
        groups.update({str(v)+"_all":h==v for v in sorted(set(h))})
        result["variants"][variant]={k:scores(data["reach"][m],data["visible"][m],
            {name:v[m] for name,v in predictions.items()}) for k,m in groups.items() if m.any()}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print({h:(s["joint"]["n"],s["joint"]["positives"],s["joint"]["auroc"]) for h,s in result["variants"]["joint_long"].items()})

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("shards",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    run(args.shards,args.output)
