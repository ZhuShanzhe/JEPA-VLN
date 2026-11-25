"""Score real detections using the trained identity head, with no instance labels."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pose_head import PoseHead,compare,conditions


def load_head(checkpoint,device="cpu"):
    ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
    head=PoseHead().to(device).eval()
    head.load_state_dict(ck["state"])
    return head,ck["mode"],float(ck["thresholds"]["detected"])


@torch.inference_mode()
def score_matrix(model,mode,left,right):
    """Frames contain C2W, independent RGB-D surface points and DINOv2 region features."""
    n,m=len(left["features"]),len(right["features"])
    score=np.full((n,m),-np.inf,dtype=np.float32)
    if not n or not m:
        return score
    lpose,rpose=np.asarray(left["c2w"]),np.asarray(right["c2w"])
    if any(t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[:3,:3].T@t[:3,:3],
           np.eye(3),atol=.005) or np.linalg.det(t[:3,:3])<.99 for t in (lpose,rpose)):
        return score
    device=next(model.parameters()).device if model is not None else "cpu"
    def array(x):
        return torch.as_tensor(np.asarray(x),dtype=torch.float32,device=device)
    valid_l=[i for i,p in enumerate(left["points"]) if p is not None and np.isfinite(p).all()]
    valid_r=[i for i,p in enumerate(right["points"]) if p is not None and np.isfinite(p).all()]
    if not valid_l or not valid_r:
        return score
    li,rj=np.meshgrid(valid_l,valid_r,indexing="ij")
    li,rj=li.ravel(),rj.ravel()
    for start in range(0,len(li),128):
        i,j=li[start:start+128],rj[start:start+128]
        a,b=array(np.asarray(left["features"])[i]),array(np.asarray(right["features"])[j])
        pa,pb=array([left["points"][k] for k in i]),array([right["points"][k] for k in j])
        ra,rb=(array(x[:3,:3]).expand(len(i),-1,-1) for x in (lpose,rpose))
        ca,cb=(array(x[:3,3]).expand(len(i),-1) for x in (lpose,rpose))
        if mode=="location":
            vals=-(pa-pb).norm(dim=1)
        elif mode=="pose_forecast":
            ab,g=conditions(ra,ca,pa,rb,cb,pb)
            ba,_=conditions(rb,cb,pb,ra,ca,pa)
            vals=model(a,b,ab,ba,torch.zeros_like(g))[0].sigmoid()
        else:
            vals=compare(model,a,b,ra,ca,pa,rb,cb,pb,mode=="heading_jepa")[0].sigmoid()
        score[i,j]=vals.cpu().numpy()
    return score


def unique_links(scores,threshold,geometric=None):
    """Retain geometrically corroborated links and veto co-visible distinct instances."""
    locks=[] if geometric is None else list(geometric["matches"])
    veto=set() if geometric is None else set(map(tuple,geometric["distinct"]))
    used_i={i for i,_ in locks}
    used_j={j for _,j in locks}
    assert len(locks)==len(used_i)==len(used_j) and not (set(locks)&veto)
    remaining_i=[i for i in range(scores.shape[0]) if i not in used_i]
    remaining_j=[j for j in range(scores.shape[1]) if j not in used_j]
    if not remaining_i or not remaining_j:
        return locks
    matrix=scores[np.ix_(remaining_i,remaining_j)]
    ok=np.isfinite(matrix)&(matrix>=threshold)
    for i,orig_i in enumerate(remaining_i):
        for j,orig_j in enumerate(remaining_j):
            if (orig_i,orig_j) in veto:
                ok[i,j]=False
    # Real zero-utility dummy columns allow directions to stay unresolved.
    value=np.zeros((len(remaining_i),len(remaining_j)+len(remaining_i)))
    value[:,:len(remaining_j)]=np.where(ok,matrix-threshold+1e-6,-1e6)
    i,j=linear_sum_assignment(-value)
    return locks+[(remaining_i[a],remaining_j[b]) for a,b in zip(i,j) if b<len(remaining_j) and ok[a,b]]
