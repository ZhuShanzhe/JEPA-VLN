"""Independently check optical outcomes against frozen-HSSD depth/instance ray-intersection labels."""
from pathlib import Path
import json
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from train_joint import ROOT,load,batch,JointPredictor,scored,binary


def run():
    saved=torch.load(ROOT/"private_train/joint_long.pt",weights_only=True,map_location="cpu")
    net=JointPredictor(saved["dims"])
    net.load_state_dict(saved["weights"])
    net.eval()
    d=load("pose")
    all_ids=np.arange(len(d["context"]))
    pred,pr,pvr,j,po=scored(net,"pose",d,all_ids,saved["stats"]["pose"].numpy(),"joint_long","cpu")
    in_frame=[]; blocked=[]; counts=[]
    for p in sorted((ROOT/"private_pose").glob("*.npz")):
        with np.load(p) as z:
            in_frame.extend(z["in_fov"]); blocked.extend(z["intercepted"]); counts.extend(z["source_point_count"])
    counts=np.asarray(counts); inside=np.asarray(in_frame); occ=np.asarray(blocked)
    assert len(inside)==len(d["context"]) and np.all((inside<0)|(occ<=inside))
    test=d["split"]=="test"; train=d["split"]=="train"
    substantial=(inside>=8)&(4*occ>=inside)
    clear=(inside>=8)&(occ==0)
    out_of_frame=inside==0
    def subgroup(mask):
        mask&=test
        vis=d["visible"][mask]
        # A source-visible patch may be occluded while another side of that instance stays visible.
        return dict(n=int(mask.sum()),houses=len(np.unique(d["house"][mask])),
                    optical_visibility=binary(vis,po[mask]) if len(vis) else None,
                    average_predicted_visibility=float(po[mask].mean()) if len(vis) else None)
    result=dict(optical_visibility_when_source_surface_blocked=subgroup(substantial.copy()),
                optical_visibility_when_source_surface_clear=subgroup(clear.copy()),
                optical_visibility_when_source_surface_off_frame=subgroup(out_of_frame.copy()))
    # Frozen latent probes. Fit on training houses and compare with a current-view-only probe.
    # At least 8 projected points limit the relative occlusion label's sampling error.
    valid=(inside>=8)&(counts>0)
    ray_fraction=occ/np.maximum(1,inside)
    mask=train&valid; held=test&valid
    for name,features in (("source_only",d["context"][:,:1024]),("predicted_endpoint",pred)):
        est=make_pipeline(StandardScaler(),Ridge(alpha=2000))
        est.fit(features[mask],ray_fraction[mask])
        prob=np.clip(est.predict(features[held]),0,1)
        result[name]=dict(train_n=int(mask.sum()),test_n=int(held.sum()),mean_fraction=float(ray_fraction[held].mean()),
                          brier_fraction=float(np.mean((prob-ray_fraction[held])**2)),
                          mae_fraction=float(np.mean(abs(prob-ray_fraction[held]))),
                          auroc_substantial=binary(substantial[held],prob)["auroc"])
    result["definition"]="Optical counterfactuals included. A sampled instance surface point is occluded when a different surface is at least 0.15 m in front on its reprojected ray. Substantial: at least 8 points in field of view and at least 25% intercepted. Labels score the probe only, never the actor."
    (ROOT/"occlusion_metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result))

if __name__=="__main__":
    torch.set_num_threads(3)
    run()
