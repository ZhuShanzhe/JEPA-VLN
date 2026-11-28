"""Check public prediction entry point and compare long panels to train-only priors."""
import json
import numpy as np
import torch
from train_joint import ROOT, binary, file_hash
from eval_pose import CameraPredictor
from train_long_cues import read_data, labels


def evaluate():
    d,masks=read_data()
    train,test=masks["train"],masks["test"]
    native=d["native_mask"][test]
    y=labels(d,test)
    prior=labels(d,train).mean()
    prior_by_steps=np.empty(len(y),dtype=float)
    for h in np.unique(d["horizon"][test]):
        pr=labels(d,train[d["horizon"][train]==h]).mean()
        prior_by_steps[d["horizon"][test]==h]=pr
    checks={}
    for name in ("long_cue_joint","long_cue_direct"):
        model=CameraPredictor(ROOT/"private_train"/(name+".pt"))
        with np.load(ROOT/"private_train"/("test_"+name+"_predictions.npz"),allow_pickle=False) as f:
            assert np.array_equal(d["source_file"][test],f["source_file"]) and np.array_equal(d["source_row"][test],f["source_row"])
            assert np.allclose(f["p_joint"],f["p_reach"]*f["p_visible_if_reach"],atol=1e-7)
            worst_roundtrip=0.
            for house in np.unique(d["house"][test]):
                hh=np.flatnonzero(d["house"][test]==house)
                ix=hh[np.linspace(0,len(hh)-1,8,dtype=int)]
                ids=test[ix]
                out=model(d["context"][ids],d["action"][ids],d["pose"][ids])
                worst_roundtrip=max(worst_roundtrip,float(np.max(abs(out["p_joint"]-f["p_joint"][ix]))))
            assert worst_roundtrip<.001, "round-trip discrepancy beyond 0.1 percentage point"
            checks[name]=dict(checkpoint=file_hash(ROOT/"private_train"/(name+".pt")),
                max_abs_joint_probability_difference=worst_roundtrip,n_cpu_roundtrip=96)
    result={"checks":checks,"prior_from_train":float(prior),"test_by_horizon":{}}
    for name,m in (("all",np.ones(len(test),bool)),("native",native)):
        result["test_by_horizon"][name]=dict(n=int(m.sum()),positive=int(y[m].sum()),positive_rate=float(y[m].mean()),
            constant_train_prior=binary(y[m],np.full(m.sum(),prior)),
            horizon_train_prior=binary(y[m],prior_by_steps[m]))
    (ROOT/"long_cue_inference_checks.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    torch.set_num_threads(3)
    evaluate()
