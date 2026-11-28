"""Check mixed-horizon inference and supported short-view proposals against the saved physical data."""
import json
from pathlib import Path
import numpy as np
import torch
from observation_head import HorizonCameraPredictor,propose

HERE=Path(__file__).resolve().parent


def check():
    torch.set_num_threads(2)
    model=HorizonCameraPredictor.from_checkpoints(HERE/"private_train")
    samples=[]
    for folder in ("private_pose","private_long_cue"):
        for path in sorted((HERE/folder).glob("test_*.npz")):
            with np.load(path,allow_pickle=False) as z:
                if not len(z["reach"]):continue
                ids=np.linspace(0,len(z["reach"])-1,7,dtype=int)
                a=z["action"][ids]
                samples.append((z["context"][ids],np.pad(a,((0,0),(0,108-a.shape[1]))),z["pose"][ids]))
    x,act,pose=(np.concatenate(b) for b in zip(*samples))
    ix=np.random.default_rng(1128).permutation(len(act))
    x,act,pose=x[ix],act[ix],pose[ix]
    probs=model(x,act,pose)
    length=(act!=0).sum(1)
    maximum=0.
    for m,sub in ((length<12,model.short),(length>=12,model.long)):
        truth=sub(x[m],act[m],pose[m])
        for key in truth:maximum=max(maximum,float(np.max(abs(probs[key][m]-truth[key]))))
    assert maximum<1e-5 and np.allclose(probs["p_joint"],probs["p_reach"]*probs["p_visible_if_reach"])
    zpath=next(iter(sorted((HERE/"private_pose").glob("test_*.npz"))))
    with np.load(zpath,allow_pickle=False) as z:
        rows=np.flatnonzero(z["group"]==z["group"][0])
        geom=z["pose"][rows]
        origin=int(np.flatnonzero(z["native_mask"][rows]&z["safe_mask"][rows])[0])
        result=propose(model,z["context"][rows[0],:1024],z["context"][rows[0],1024:],
                       z["action"][rows],geom,z["safe_mask"][rows],origin)
        k=result["index"]
        assert z["native_mask"][rows[k]] and z["safe_mask"][rows[k]]
        assert len(result["action"])>0 and result["predicted_future"].shape==(1024,)
    report={"mixed_batch_native_checkpoint_max_error":maximum,"short_samples":int((length<12).sum()),
            "long_samples":int((length>=12).sum()),"actual_fixed_camera_decision_checked":True}
    (HERE/"head_acceptance.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)

if __name__=="__main__":check()
