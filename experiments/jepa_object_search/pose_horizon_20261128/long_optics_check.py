"""Hold body and route fixed and choose among three sensor settings at a hypothetical endpoint."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from eval_pose import CameraPredictor

HERE=Path(__file__).resolve().parent


def audit():
    torch.set_num_threads(2)
    result={}
    for variant in ("joint_long","long_cue_joint","long_cue_direct"):
        head=CameraPredictor(HERE/"private_train"/(variant+".pt"))
        counts=Counter()
        for shard in sorted((HERE/"private_long_cue").glob("test_*.npz")):
            with np.load(shard,allow_pickle=False) as z:
                ctx,a,x=z["context"],z["action"],z["pose"]
                if not len(x):continue
                j=np.concatenate([head(ctx[ix],a[ix],x[ix])["p_joint"] for ix in
                                  np.array_split(np.arange(len(ctx)),max(1,(len(ctx)+63)//64))])
                label=(z["reach"]==1)&(z["visible"]==1)
                groups=z["group"]
                pid=z["plan_id"]
                native=z["native_mask"]
                for source in np.unique(groups):
                    for plan in np.unique(pid[groups==source]):
                        ids=np.flatnonzero((groups==source)&(pid==plan))
                        assert len(ids)==3 and len(np.unique(x[ids,:2],axis=0))==1
                        current,=ids[native[ids]]
                        best=ids[np.argmax(j[ids])]
                        changed=best if j[current]<.5 and j[best]>=.5 and j[best]-j[current]>=.05 else current
                        counts.update(groups=1,native=int(label[current]),best_available=int(label[ids].any()),
                                      predicted_highest=int(label[best]),confidence_gate=int(label[changed]),
                                      rescued_confidence_gate=int(label[changed] and not label[current]),
                                      spoiled_confidence_gate=int(not label[changed] and label[current]))
        result[variant]=dict(counts)
    report=dict(method="Same cue, already chosen action path and endpoint for all three views. The articulating sensor rig is hypothetical. Ties go to earliest offered view. Both raw and confidence-gated selections ignore evaluator truth.",variants=result)
    (HERE/"long_optics.json").write_text(json.dumps(report,indent=2)+"\n")
    print(result)

if __name__=="__main__":audit()
