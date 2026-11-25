"""Condition a frozen region representation on the change in camera heading."""
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F


SIZE = 224
DIM = 2304
OBJECT_DIM = 1536
REPRESENTATION = "dinov2_vitb14_cls_patchmean_maskedcontext_v1"


def mask_context(rgb, box_in_context):
    image=np.array(rgb,dtype=np.uint8,copy=True)
    x0,y0,x1,y1=map(int,box_in_context)
    if not (0<=x0<x1<=image.shape[1] and 0<=y0<y1<=image.shape[0]):
        raise ValueError("Invalid object box in contextual crop")
    image[y0:y1,x0:x1]=[124,116,104]
    return image


def preprocess(rgb):
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    image = image.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.
    array = (array - np.array([.485,.456,.406],dtype=np.float32)) / np.array([.229,.224,.225],dtype=np.float32)
    return torch.from_numpy(array.transpose(2,0,1).copy())


class Encoder(nn.Module):
    def __init__(self, project):
        super().__init__()
        hub = Path(project)/"mapping_runtime/weights/torch/hub"
        self.base = torch.hub.load(str(hub/"facebookresearch_dinov2_main"),"dinov2_vitb14",source="local",pretrained=False)
        state = torch.load(hub/"checkpoints/dinov2_vitb14_pretrain.pth",map_location="cpu",weights_only=True)
        self.base.load_state_dict(state,strict=True)
        self.base.requires_grad_(False).eval()

    @torch.inference_mode()
    def forward(self, images, masked_contexts):
        features = self.base.forward_features(images)
        cls = F.normalize(features["x_norm_clstoken"].float(),dim=-1)
        mean = F.normalize(features["x_norm_patchtokens"].float().mean(1),dim=-1)
        context=self.base.forward_features(masked_contexts)["x_norm_clstoken"].float()
        return torch.cat([math.sqrt(.425)*cls,math.sqrt(.425)*mean,
                          math.sqrt(.15)*F.normalize(context,dim=-1)],dim=-1)


class ViewHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = nn.Linear(DIM,128)
        self.predict = nn.Sequential(nn.Linear(132,128),nn.GELU(),nn.Linear(128,DIM))
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)
        self.log_scale = nn.Parameter(torch.tensor(math.log(18.)))
        self.bias = nn.Parameter(torch.tensor(-12.))

    def forecast(self, anchor, degrees):
        theta=torch.deg2rad(degrees)
        angle=torch.stack([theta.sin(),theta.cos(),(2*theta).sin(),(2*theta).cos()],dim=-1)
        local=F.gelu(self.compress(anchor*math.sqrt(DIM)))
        residual=self.predict(torch.cat([local,angle],dim=-1))/math.sqrt(DIM)
        return F.normalize(anchor+residual,dim=-1)

    def forward(self, anchor, candidate, degrees):
        future=self.forecast(anchor,degrees)
        reverse=self.forecast(candidate,-degrees)
        sim=((future*candidate).sum(-1)+(reverse*anchor).sum(-1))*.5
        logit=self.log_scale.exp().clamp(1,64)*sim+self.bias
        return logit,sim


class Matcher:
    @torch.inference_mode()
    def compare(self, reference_rgb, candidate_rgb, yaw_delta_deg, *, reference_context, candidate_context):
        if not math.isfinite(float(yaw_delta_deg)):
            raise ValueError("Both observations require a finite camera heading")
        feat=self.encoder(torch.stack([preprocess(reference_rgb),preprocess(candidate_rgb)]).to(self.device),
                          torch.stack([preprocess(reference_context),preprocess(candidate_context)]).to(self.device))
        score=self.head(feat[:1],feat[1:],torch.tensor([float(yaw_delta_deg)],device=self.device))[0].sigmoid().item()
        return dict(score=score,threshold=self.threshold,match=score>=self.threshold)


def load_matcher(project, weights, device="cpu"):
    checkpoint=torch.load(weights,map_location="cpu",weights_only=True)
    assert checkpoint["model"]=="jepa" and checkpoint["representation"]==REPRESENTATION
    api=Matcher()
    api.device=device
    api.encoder=Encoder(project).to(device).eval()
    api.head=ViewHead()
    api.head.load_state_dict(checkpoint["state_dict"])
    api.head.to(device).eval()
    api.threshold=float(checkpoint["threshold"])
    return api
