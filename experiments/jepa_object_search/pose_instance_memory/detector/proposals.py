"""RGB-only class proposals. Only the subsequent identity resolver can register an object."""
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image


CAPTIONS = {"wall_art":"painting", "picture_frame":"picture frame",
            "wall_clock":"clock", "alarm_clock":"clock", "mantel_clock":"clock",
            "grandfather_clock":"clock", "television_receiver":"television"}
MODES = ("base","empty","augment")


def iou(a,b):
    w = max(0.,min(a[2],b[2])-max(a[0],b[0]))
    h = max(0.,min(a[3],b[3])-max(a[1],b[1]))
    inter = w*h
    area = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/area if area > 0 else 0.


def select(raw,mode):
    if mode not in MODES:
        raise ValueError("Unknown proposal mode")
    base = [b for b in raw if b["view"] == "full"]
    candidates = base if mode=="base" or (mode=="empty" and base) else raw
    kept = []
    for b in sorted(candidates,key=lambda r:-r["score"]):
        if not any(iou(b["xyxy"],k["xyxy"]) >= .85 for k in kept):
            kept.append(b)
    return kept


def configured(project):
    p = Path(project)/"jepa_reid_v21/detector/protocol.json"
    if not p.exists():
        return "base"
    mode = json.loads(p.read_text())["selected_mode"]
    if mode not in MODES:
        raise ValueError("Invalid calibrated protocol")
    return mode


class LocalCandidateDetector:
    def __init__(self, project, device="cpu", mode=None):
        import torch
        self.mode = mode or configured(project)
        self.device = device
        self.project = Path(project).resolve()
        detector = self.project/"b_verification/detector_run"
        for key in ("TRANSFORMERS_OFFLINE","HF_HUB_OFFLINE"):
            os.environ[key] = "1"
        sys.path.insert(0,str(detector))
        from groundingdino.util.inference import load_model,predict
        import groundingdino.datasets.transforms as T
        if device.startswith("cuda"):
            # The official PyTorch implementation also used by this installation
            # avoids an unavailable custom CUDA operator on the RTX 5090.
            from groundingdino.models.GroundingDINO import ms_deform_attn as ms
            ms.MultiScaleDeformableAttnFunction.apply = lambda v,s,ix,loc,w,step: ms.multi_scale_deformable_attn_pytorch(v,s,loc,w)
        torch.set_num_threads(3)
        cwd = os.getcwd()
        try:
            os.chdir(detector)
            self.model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py",
                                    "weights/groundingdino_swint_ogc.pth",device=device).eval().to(device)
        finally:
            os.chdir(cwd)
        self.predict = predict
        self.transform = T.Compose([T.RandomResize([800],max_size=1333),T.ToTensor(),
                                    T.Normalize([.485,.456,.406],[.229,.224,.225])])

    def infer_view(self,rgb,caption,crop,view):
        x0,y0,x1,y1=crop
        h,w=rgb.shape[:2]
        im=Image.fromarray(rgb[y0:y1,x0:x1])
        image,_=self.transform(im,None)
        boxes,scores,labels=self.predict(self.model,image,caption,.35,.25,device=self.device)
        result=[]
        for (x,y,bw,bh),score,label in zip(boxes.tolist(),scores.tolist(),labels):
            rect=np.clip([x-bw/2,y-bh/2,x+bw/2,y+bh/2],0,1)
            xa,ya,xb,yb=rect*np.asarray([x1-x0,y1-y0,x1-x0,y1-y0])+[x0,y0,x0,y0]
            if xb-xa<2 or yb-ya<2:
                continue
            if view!="full" and ((x0>0 and xa<=x0+3) or (x1<w and xb>=x1-3)
                             or (y0>0 and ya<=y0+3) or (y1<h and yb>=y1-3)):
                continue
            result.append(dict(xyxy=list(map(float,(xa,ya,xb,yb))),score=float(score),phrase=label,
                               source="groundingdino",view=view))
        return result

    def proposals(self,rgb,category,mode=None):
        mode=mode or self.mode
        rgb=np.asarray(rgb)
        if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:
            raise ValueError("Requires an RGB uint8 image")
        key=str(category).strip().lower().rstrip(".")
        if not key or len(key)>128:
            raise ValueError("Missing or unsupported goal description")
        caption=CAPTIONS.get(key,key.replace("_"," "))+"."
        h,w=rgb.shape[:2]
        raw=self.infer_view(rgb,caption,(0,0,w,h),"full")
        if mode=="augment" or (mode=="empty" and not raw):
            sw,sh=max(2,round(w*.8)),max(2,round(h*.8))
            top=(h-sh)//2
            for left,v in ((0,"left"),(w-sw,"right")):
                raw.extend(self.infer_view(rgb,caption,(left,top,left+sw,top+sh),v))
        return raw

    def detect(self,rgb,category):
        raw=self.proposals(rgb,category)
        return [dict(row,bbox=row["xyxy"]) for row in select(raw,self.mode)]
