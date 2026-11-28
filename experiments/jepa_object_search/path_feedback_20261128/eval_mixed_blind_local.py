"""Apply both frozen path checkers to the six additional houses' seven-way target-preserving detours."""
from pathlib import Path
import hashlib,json,os,sys
import numpy as np
from threadpoolctl import threadpool_limits
from checker import PathChecker
from mixed_checker import MixedPathChecker
from train_checker import classification,score_prior
from train_mixed_checker import select_report

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/"pose_horizon_20261128"
sys.path.insert(0,str(PRIOR))
from observation_head import HorizonCameraPredictor


def main():
    import torch
    os.umask(0o077)
    torch.set_num_threads(2)
    old=HorizonCameraPredictor.from_checkpoints(PRIOR/"private_train","cuda" if torch.cuda.is_available() else "cpu")
    mpath=HERE/"private_checker_mixed/checker.joblib"
    sums=[hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/"private_checker/checker.joblib",mpath)]
    assert sums[1]==json.loads((HERE/"mixed_results.json").read_text())["frozen_before_test_sha256"]
    checkers=[PathChecker.load(HERE/"private_checker/checker.joblib"),MixedPathChecker.load(mpath)]
    dirs=[HERE/"private_blind_checker",HERE/"private_blind_checker_mixed"]
    for d in dirs:d.mkdir(exist_ok=True)
    files=sorted(p for p in (HERE/"private_blind_paired").glob("blind_*.npz") if ".partial." not in p.name)
    assert len(files)==6
    nat={p.name for p in (HERE/"private_natural").glob("blind_*.npz")}
    assert {p.name for p in files}==nat
    observations=[]
    for path in files:
        with np.load(path,allow_pickle=False) as d:
            a,x=d["action"],d["path"]
            y=d["reach"].astype(bool)
            prev=score_prior(d,old)
            outs=[]
            for folder,model in zip(dirs,checkers):
                p=model.predict(a,x)
                np.savez_compressed(folder/path.name,group=d["group"],plan_id=d["plan_id"],p_old=prev,**p)
                outs.append(p)
            observations.append((y,prev,outs[0]["p_reach"],outs[0]["threshold"],
                                 outs[1]["p_reach"],outs[1]["threshold"],outs[1]["screening"],
                                 d["safe_mask"],((x[:,:,13]>.5)&(a==1)).any(1)))
    y,prev,v1,t1,v2,t2,s,safe,hit=(np.concatenate(v) for v in zip(*observations))
    result=dict(weight_sha256=sums,test_houses=6,original_jepa_half=classification(y,prev,.5),
                original_safety_veto=classification(y,safe.astype(float),.5),
                points_only=classification(y,(~hit).astype(float),.5),
                checker_v1=classification(y,v1,t1),mixed=classification(y,v2,t2),selective=select_report(y,s))
    (HERE/"checker_blind_local.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result))

if __name__=="__main__":
    with threadpool_limits(limits=3):main()
