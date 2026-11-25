"""Verify camera-frame invariance, exchange symmetry and use of an observed location."""
import argparse
import json
from pathlib import Path

import torch
from pose_head import compare
from pose_inference import load_head
from train_pose import load


@torch.inference_mode()
def main(project):
    out=project/"jepa_reid_v21/model"
    data,sets,kinds,_=load(project)
    f,R,C,P=data
    pairs=[(i,j) for i,j,y,kind,house,c in kinds["test"] if kind=="adjacent" and y==1]
    a,b=zip(*pairs)
    a,b=list(a),list(b)
    head,_,_=load_head(out/"pose_jepa.pt","cuda")
    ra,ca,pa,rb,cb,pb=R[a],C[a],P[a],R[b],C[b],P[b]
    def score(feat1,feat2,*pose):
        return compare(head,feat1,feat2,*pose)[0].sigmoid()
    base=score(f[a],f[b],ra,ca,pa,rb,cb,pb)
    reverse=score(f[b],f[a],rb,cb,pb,ra,ca,pa)
    g=R[a[0]].T.contiguous()
    t=ca.new_tensor([2.,-.8,3.])
    transform=lambda x:x @ g.T+t
    changed_frame=score(f[a],f[b],g@ra,transform(ca),transform(pa),g@rb,transform(cb),transform(pb))
    offset=rb[:,:,0] # One simulated metre to the query camera's right, with RGB frozen.
    shifted=score(f[a],f[b],ra,ca,pa,rb,cb,pb+offset)
    symmetry=(base-reverse).abs().max().item()
    invariance=(base-changed_frame).abs().max().item()
    assert symmetry<1e-4 and invariance<1e-4
    audit={"test_positive_pairs":len(pairs),"max_swap_probability_error":symmetry,
        "max_rigid_frame_change_probability_error":invariance,"location_shift_metres":1.,
        "pairs_with_lower_score_after_location_conflict":int((shifted<base).sum().item()),
        "median_probability_change":(shifted-base).median().item(),"passed":True}
    (out/"pose_input_checks.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("project",type=Path)
    main(parser.parse_args().project)
