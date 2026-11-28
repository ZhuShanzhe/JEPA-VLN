"""Paired offline safety diagnostic. Locked model and thresholds, unchanged navigation code."""
import json
from pathlib import Path
import numpy as np
import torch
from prior_cues_score import HERE,visual_input,FrozenImageEncoder,CameraPredictor,correct_view

def main():
    torch.set_num_threads(2)
    predictor=CameraPredictor(HERE/"private_train/joint_long.pt")
    assert predictor.variant=="joint_long"
    encoder=FrozenImageEncoder()
    arms={"single":[], "three_clearance_only":[], "three_matching_features": []}
    lamp={}
    files=sorted((HERE/"private_prior_safety").glob("*.npz"))
    assert len(files)==4
    for f in files:
        with np.load(f) as meas,np.load(HERE/"private_prior_cues"/f.name) as d:
            enc=encoder.encode([visual_input(d["rgb"]),visual_input(d["crop"])])
            ctx=np.tile(enc.ravel(),(9,1))
            original=d["pose"]
            success=d["reach"] & (d["pixels"]>=369)
            if f.stem=="1_1":
                acceptable=np.flatnonzero(success)
                assert len(acceptable)==1
                i=int(acceptable[0])
                lamp=dict(source_pixels=int(d["source_pixels"]),four_forward_pixels=int(d["pixels"][i]),
                          path_samples=48,target="table lamp",action=d["action"][i,:4].tolist(),
                          single_observed=float(meas["single"][i,0]),three_observed=float(meas["three"][i,0]),
                          single_hazard=float(meas["single"][i,2]),three_hazard=float(meas["three"][i,2]))
            for arm,records in arms.items():
                coverage=meas["single" if arm=="single" else "three"]
                feat=original.copy()
                if arm=="three_matching_features":feat[:,-2:]=coverage[:,:2]
                out=predictor(ctx,d["action"],feat)
                safe=coverage[:,3].astype(bool)
                idx=correct_view(feat,out["p_reach"],out["p_joint"],safe,0)
                records.append(dict(successful_cues=int(idx is not None and success[idx]),
                    successful_alternative_available=int(np.any(safe&success)),
                    changed_from_source=int(idx is not None and idx!=0),safe_candidates=int(safe.sum()),
                    safely_admitted_successful_candidates=int((safe & success).sum())))
                if f.stem=="1_1":
                    lamp[arm]=dict(candidate_passed=bool(safe[i]),candidate_p_joint=float(out["p_joint"][i]),
                        candidate_p_reach=float(out["p_reach"][i]),
                        candidate_p_visible_if_reach=float(out["p_visible_if_reach"][i]),
                        source_p_joint=float(out["p_joint"][0]),chosen_candidate=int(idx) if idx is not None else None,
                        chose_only_success=int(idx==i),candidate=i)
    summary={arm:{k:sum(r[k] for r in rows) for k in rows[0]} for arm,rows in arms.items()}
    prior=json.loads((HERE/"prior_cues.json").read_text())
    assert summary["single"]["successful_cues"]==prior["chosen_joint"]
    report=dict(cues=4,houses=3,candidate_plans=36,reachability_and_same_instance_threshold="unchanged, >=1%",
        fixed="Original three side/center RGB-D hashes checked before scoring. All three were acquired within the original four scan actions. Same pose checkpoint, candidate plans and 65% observed-path threshold. A hazard from any observed ray disqualifies that candidate.",
        analyses="three_clearance_only substitutes multi-view safety mask, preserving the exact head inputs. three_matching_features additionally updates the last two input path-coverage fractions, keeping the same head weights.",
        model_sha256=prior["model_sha256"],results=summary,lamp=lamp)
    (HERE/"historical_safety.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
