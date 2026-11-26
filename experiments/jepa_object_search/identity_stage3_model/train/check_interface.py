"""Confirm that runtime images reproduce stored development features and tri-state decisions."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from identity import IdentityVerifier,encode_frame_from_features


def run(project,output):
    root=project/"identity_stage3_model/train/private"
    cache=torch.load(root/"val.pt",weights_only=True,map_location="cpu")
    frames={r["frame_id"]:r for r in map(json.loads,(root/"val/frames.jsonl").read_text().splitlines())}
    pred={r["frame_id"]:r["boxes"] for r in map(json.loads,(root/"val/det/predictions.jsonl").read_text().splitlines())}
    observed=[]
    gtvec=[]
    errors=[]
    v=IdentityVerifier(project)
    for index,rec in enumerate(cache["records"]):
        fid=rec["frame"]
        if rec["jitter"] or not rec["instance"] or any(prev==fid for prev,_,_ in observed):
            continue
        fr=frames[fid]
        assert fr["split"]=="val"
        rgb=np.asarray(Image.open(root/"val"/fr["rgb"]).convert("RGB"))
        xyxy=[pred[fid][rec["slot"]]["xyxy"]]
        frame=v.encode(rgb,xyxy,np.load(root/"val"/fr["depth"]),fr["K"],fr["T_world_from_camera"])
        delta=float(np.max(np.abs(frame["features"][0]-cache["features"][index].numpy())))
        point=float(np.linalg.norm(frame["points"][0]-rec["point"]))
        assert delta<1e-4 and point<1e-5, (delta,point)
        errors.append((delta,point))
        observed.append((fid,frame,rec))
        gtvec.append(encode_frame_from_features(cache["features"][index:index+1].numpy(),[rec["point"]],fr["T_world_from_camera"]))
        if len(observed)==3:
            break
    assert len(observed)==3
    score_errors={}
    for mode in ("jepa","direct"):
        v=IdentityVerifier(project,mode)
        actual=v.score(observed[0][1],observed[1][1])
        cached=v.score(gtvec[0],gtvec[1])
        gap=float(np.max(np.abs(actual-cached)))
        assert gap<1e-4
        assert v.decide(np.array([[1.,0.]]))[0]==dict(state="same",index=0)
        assert v.decide(np.array([[1.,1.]]))[0]["state"]=="uncertain"
        assert v.decide(np.array([[0.]]))[0]["state"]=="different"
        assert v.decide(np.array([[np.nan]]))[0]["state"]=="uncertain"
        assert v.decide(np.empty((1,0)))[0]["state"]=="uncertain"
        assert all(r["state"]=="uncertain" for r in v.decide(np.ones((2,1))))
        no_pose=v.encode(rgb,xyxy,None,None,None)
        assert np.isnan(v.score(no_pose,observed[0][1])).all()
        score_errors[mode]=gap
    output.write_text(json.dumps(dict(real_calibration_images=len(observed),
        max_abs_region_encoding_difference=max(v[0] for v in errors),
        max_metric_support_difference_m=max(v[1] for v in errors),
        max_abs_cached_vs_online_pair_score_difference=score_errors,
        checks_per_method=7,paid_calls=0),indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("project",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    run(args.project.resolve(),args.output)
