"""Cross-view identity evidence from observable boxes, camera transforms and depth."""
import math

import torch
from torch import nn
import torch.nn.functional as F


DIM=2304
POSE_DIM=15
GEO_DIM=5


def local(rotation, displacement):
    return (rotation.transpose(1,2) @ displacement.unsqueeze(-1)).squeeze(-1)


def signed_log(x):
    return x.sign()*torch.log1p(x.abs().clamp(max=50))


def conditions(ra,ca,pa,rb,cb,pb,heading_only=False):
    # OpenCV camera-to-world convention. The forward predictor does not see pb,
    # the target crop, a world coordinate, or an object/scene label.
    relative=ra.transpose(1,2) @ rb
    translation=local(ra,cb-ca)
    anchor=local(ra,pa-ca)
    if heading_only:
        angle=torch.atan2(relative[:,0,2],relative[:,2,2])
        pose=ra.new_zeros((len(ra),POSE_DIM))
        pose[:,0]=angle.sin()
        pose[:,1]=angle.cos()
        return pose,ra.new_zeros((len(ra),GEO_DIM))
    pose=torch.cat([relative[:,:,:2].reshape(-1,6),signed_log(translation),signed_log(anchor),
                    F.normalize(anchor,dim=1)],dim=1)
    # Only after prediction: compare independently observed support points.
    delta=pb-pa
    rad_a=F.cosine_similarity(pa-ca,pb-ca,dim=1).clamp(-1,1)
    rad_b=F.cosine_similarity(pa-cb,pb-cb,dim=1).clamp(-1,1)
    za=local(ra,delta)[:,2].abs()
    zb=local(rb,delta)[:,2].abs()
    geom=torch.stack([torch.log1p(delta.norm(dim=1)),torch.log1p((za+zb)*.5),
                      torch.sqrt((1-rad_a).clamp_min(0)),torch.sqrt((1-rad_b).clamp_min(0)),
                      torch.log1p((cb-ca).norm(dim=1))],dim=1)
    return pose,geom


class PoseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.source=nn.Linear(DIM,128)
        self.predict=nn.Sequential(nn.Linear(128+POSE_DIM,128),nn.GELU(),nn.Linear(128,DIM))
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)
        self.verify=nn.Sequential(nn.Linear(GEO_DIM+4,32),nn.GELU(),nn.Linear(32,1))

    def forecast(self,source,pose):
        ctx=F.gelu(self.source(source*math.sqrt(DIM)))
        proposal=self.predict(torch.cat([ctx,pose],dim=1))/math.sqrt(DIM)
        return F.normalize(source+proposal,dim=1)

    def forward(self,a,b,pose_ab,pose_ba,geo):
        f=self.forecast(a,pose_ab)
        r=self.forecast(b,pose_ba)
        sim=((f*b).sum(1)+(r*a).sum(1))*.5
        obj=(F.cosine_similarity(f[:,:1536],b[:,:1536])+F.cosine_similarity(r[:,:1536],a[:,:1536]))*.5
        ctx=(F.cosine_similarity(f[:,1536:],b[:,1536:])+F.cosine_similarity(r[:,1536:],a[:,1536:]))*.5
        evidence=torch.cat([torch.stack([sim,(a*b).sum(1),obj,ctx],dim=1),geo],dim=1)
        return 15*(sim-.75)+self.verify(evidence).squeeze(1),sim


def compare(head,a,b,ra,ca,pa,rb,cb,pb,heading_only=False):
    ab,g=conditions(ra,ca,pa,rb,cb,pb,heading_only)
    ba,h=conditions(rb,cb,pb,ra,ca,pa,heading_only)
    return head(a,b,ab,ba,(g+h)*.5)
