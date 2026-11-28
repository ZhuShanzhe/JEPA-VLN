"""Inference round-trip, held-out paired differences, and body-reach invariance."""
import json
from pathlib import Path
import numpy as np
import torch
from eval_pose import CameraPredictor
from train_joint import ROOT,file_hash


def main():
    folder=ROOT/"private_train"
    variants=json.loads((folder/"metrics.json").read_text())["variants"]
    predictions={k:np.load(folder/("test_pose_predictions.npz" if k=="joint_long" else f"test_pose_{k}.npz"))
                 for k in variants}
    main_pred=predictions["joint_long"]
    checkpoint=folder/"joint_long.pt"
    model=CameraPredictor(checkpoint)
    roundtrip=0
    max_roundtrip_error=0.
    invariance=0.
    for name in np.unique(main_pred["source_file"]):
        m=(main_pred["source_file"]==name)
        original=main_pred["source_row"][m]
        with np.load(ROOT/"private_pose"/name) as data:
            rows=original[np.linspace(0,len(original)-1,8,dtype=int)]
            out=model(data["context"][rows],data["action"][rows],data["pose"][rows])
            idx=np.flatnonzero(m)[np.linspace(0,len(original)-1,8,dtype=int)]
            max_roundtrip_error=max(max_roundtrip_error,float(np.max(abs(out["p_joint"]-main_pred["p_joint"][idx]))))
            roundtrip+=len(rows)
            groups=np.asarray([f"{g}:{p}" for g,p in zip(data["group"][original],data["plan_id"][original])])
            for g in np.unique(groups):
                ids=groups==g
                assert ids.sum()==9
                invariance=max(invariance,float(np.ptp(main_pred["p_reach"][m][ids])))
    print("cpu gpu maximum joint probability delta",max_roundtrip_error,flush=True)
    assert max_roundtrip_error<1e-4 and invariance<1e-5
    assert np.max(np.abs(main_pred["p_reach"]*main_pred["p_visible_if_reach"]-main_pred["p_joint"]))<1e-6
    house=np.unique(main_pred["house"])
    rng=np.random.default_rng(1128)
    boot=rng.integers(0,len(house),size=(4000,len(house)))
    paired={}
    for k,z in predictions.items():
        if k=="joint_long": continue
        assert np.array_equal(main_pred["source_file"],z["source_file"]) and np.array_equal(main_pred["source_row"],z["source_row"])
        y=main_pred["reach"]*(main_pred["visible"]>0)
        differences=(main_pred["p_joint"]-y)**2-(z["p_joint"]-y)**2
        vals=np.array([differences[main_pred["house"]==h].mean() for h in house])
        paired[k]=dict(joint_minus_control_brier=float(vals.mean()),houses_better=int((vals<0).sum()),
                       houses=len(house),house_bootstrap_95=np.quantile(vals[boot].mean(1),[.025,.975]).tolist())
    report=dict(inference_roundtrip_rows=roundtrip,roundtrip_cpu_gpu_max_abs_diff=max_roundtrip_error,body_reach_max_delta_across_9_optics=invariance,
                checkpoint_hashes={k:file_hash(folder/(k+".pt")) for k in variants},paired=paired)
    (ROOT/"joint_inference_checks.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report))

if __name__=="__main__":
    torch.set_num_threads(3)
    main()
