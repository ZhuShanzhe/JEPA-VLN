"""Independent real-box external diagnostic at original sensor extrinsics, without API calls."""
from io import BytesIO
from hashlib import sha256
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from long_data import FrozenImageEncoder
from eval_pose import CameraPredictor
from refine_view import correct_view

HERE=Path(__file__).resolve().parent


def visual_input(image):
    # Match JPEG storage and frozen preprocessing from the training image pipeline.
    data=BytesIO()
    Image.fromarray(image).save(data,format="JPEG",quality=92)
    data.seek(0)
    with Image.open(data) as im:return np.asarray(im.convert("RGB"))


def main():
    torch.set_num_threads(2)
    checkpoint=HERE/"private_train/joint_long.pt"
    model=CameraPredictor(checkpoint)
    assert model.variant=="joint_long"
    encoder=FrozenImageEncoder()
    preds=[]
    blocked_alternatives=[]
    files=sorted((HERE/"private_prior_cues").glob("*.npz"))
    assert len(files)==4
    for p in files:
        with np.load(p) as d:
            emb=encoder.encode([visual_input(d["rgb"]),visual_input(d["crop"])])
            assert d["action"].shape==(9,12) and np.allclose(d["pose"][:,2],1.31) and np.allclose(d["pose"][:,6],0.)
            pred=model(np.tile(emb.flatten(),(9,1)),d["action"],d["pose"])
            prob=pred["p_joint"]
            idx=correct_view(d["pose"],pred["p_reach"],prob,d["safe"],0)
            reachable=d["reach"]
            visible=d["pixels"]>=math.ceil(.01*192*192)
            success=reachable & visible
            if not success[0] and success.any() and not (success & d["safe"]).any():
                blocked_alternatives.append(dict(source_visible_pixels=int(d["source_pixels"]),
                    acceptable_candidates_pre_clearance=[
                        dict(action=d["action"][i][d["action"][i]!=0].tolist(),
                             reached_same_instance_pixels=int(d["pixels"][i]),
                             path_observed_fraction=float(d["pose"][i,-2]),
                             path_observed_and_clear_fraction=float(d["pose"][i,-1]))
                        for i in np.flatnonzero(success)]))
            preds.append(dict(initial_joint=bool(success[0]),chosen_joint=bool(idx is not None and success[idx]),
                              changed=idx is not None and idx!=0,unavailable=idx is None,
                              better=bool(idx is not None and success[idx] and not success[0]),
                              worse=bool(success[0] and not (idx is not None and success[idx])),
                              oracle=bool(success.any()),safe_oracle=bool((success & d["safe"]).any()),
                              source_already_clear=bool(d["source_pixels"]>=math.ceil(.01*192*192)),
                              reached=int(reachable.sum()),joint=int(success.sum()),safe=int(d["safe"].sum()),
                              safe_joint=int((d["safe"]&success).sum()),
                              max_reset_rgb_mae=float(d["rgb_reset_mae"]),max_reset_depth_mae=float(d["depth_reset_mae"]),
                              brier=float(np.mean((prob-success.astype(float))**2)),
                              source_sha_exact=bool(d["source_sha_exact"])))
    totals={k:sum(x[k] for x in preds) for k in ("initial_joint","chosen_joint","changed","unavailable","better",
              "worse","oracle","safe_oracle","source_already_clear","reached","joint","safe","safe_joint","source_sha_exact")}
    report=dict(source="Historical HSSD-OV side detections, selected before scoring using valid_box && anchored && !site_offered from interface_review.json; all four events across three houses.",
                houses=3,cues=len(files),candidates=len(files)*9,source_and_scan_rgbd_sha="All three original observed views of each scan matched RGB and depth hashes; actor consumed only selected side RGB/depth and actual detector box.",
                policy="Same nine predetermined native ground action plans for all cues. First plan turns left and right back to its original view. Height 1.31 m, horizontal, 30 degree turns, 0.25 m forward. Clearance uses only source depth. Same default correction thresholds as the head.",
                physical_labels="Execute on historical navigation simulator, reachable iff every forward action travels >=0.23 m and endpoint within 0.14 m of the request. At the actual endpoint the same instance must cover >=1% of full image (369/36864 pixels). Box association and instance pixels evaluator only.",
                model_sha256=sha256(checkpoint.read_bytes()).hexdigest(),**totals,
                highest_source_reset_rgb_mean_absolute_uint8=max(x["max_reset_rgb_mae"] for x in preds),
                highest_source_reset_depth_mean_absolute_m=max(x["max_reset_depth_mae"] for x in preds),
                candidate_joint_brier=sum(x["brier"] for x in preds)/len(preds),
                clearance_rejected_summary=blocked_alternatives,
                scope="Frozen local counterfactual with 4 cues, including 2 cues from one house. No full navigation episode or distinct-report metric.")
    (HERE/"prior_cues.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
