"""Serialize row-aligned actor predictions using only runtime observations and candidate actions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from view_model import ViewPredictor

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"path_feedback_20261128"))
sys.path.insert(0,str(HERE.parent/"pose_horizon_20261128"))
from mixed_checker import MixedPathChecker
from observation_head import HorizonCameraPredictor


def main(split):
    torch.set_num_threads(3)
    os.umask(0o077)
    frozen=json.loads((HERE/"view_training.json").read_text())["models"]
    heads={}
    for name,meta in frozen.items():
        file=HERE/"private_model"/(name+".pt")
        assert hashlib.sha256(file.read_bytes()).hexdigest()==meta["sha256"]
        heads[name]=ViewPredictor.load(file)
    weights=HERE.parent/"path_feedback_20261128"/"private_checker_mixed"/"checker.joblib"
    assert hashlib.sha256(weights.read_bytes()).hexdigest()=="b20520e5c387584d74fa400b009b64f6cfc2c3ff49fd688c37f9e23536980c41"
    checker=MixedPathChecker.load(weights)
    old=HorizonCameraPredictor.from_checkpoints(HERE.parent/"pose_horizon_20261128"/"private_train","cuda" if torch.cuda.is_available() else "cpu")
    dest=HERE/"private_predictions";dest.mkdir(exist_ok=True)
    files=[p for p in sorted((HERE/"private_data").glob(split+"_*.npz")) if ".partial" not in p.name]
    expected=json.loads((HERE/"view_design.json").read_text())["cohorts"][split]
    if {p.stem for p in files}!={split+"_"+h for h in expected}:
        raise ValueError("Incomplete or unexpected house split: "+split)
    for file in files:
        with np.load(file,allow_pickle=False) as d:
            # Metadata preserved solely for row alignment. No episode ID or labels fed to any model.
            rows={k:d[k] for k in ("group","view_id","plan_id","kind")}
            context,pose,mp,action,path=[d[k] for k in ("context","pose","map","action","path")]
        for name,head in heads.items():
            rows["p_visible_"+name]=head.predict(context,pose,mp)["p_visible"]
        rows.update(checker.predict(action,path))
        old_visible=[]
        for i in range(0,len(pose),128):
            old_visible.extend(old(context[i:i+128],action[i:i+128],pose[i:i+128])["p_visible_if_reach"])
        rows["p_visible_frozen"]=np.asarray(old_visible)
        temp=(dest/file.name).with_suffix(".partial.npz")
        np.savez_compressed(temp,**rows)
        temp.replace(dest/file.name)
        print(file.stem,len(pose),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("split",choices=("train","val","blind"))
    with threadpool_limits(limits=3):main(parser.parse_args().split)
