"""Frozen local semantic ranking shared by every source of boxes."""
import re
from pathlib import Path

import numpy as np


NAMES = ("chair","painting","mirror","stool","clock","bookcase","picture frame","vase",
         "table","sofa","ceiling lamp","cabinet","basket","bench","potted plant","bed")
BACKGROUND = ("an empty wall","the floor","the ceiling","a door","a window","an empty corner")


def noun(text):
    phrase = str(text).lower().replace("_"," ").strip()
    if re.search(r"\b(clocks?|timepiece)\b",phrase):
        return "clock"
    if re.search(r"\b(paintings?|artworks?|wall art|landscape)\b",phrase):
        return "painting"
    return phrase


class RegionRanker:
    def __init__(self, project, device="cpu"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device
        path = Path(project)/"comparisons_nov14/weights/openai--clip-vit-large-patch14"
        self.model = CLIPModel.from_pretrained(path,local_files_only=True).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(path,local_files_only=True)
        self.text_cache = {}

    def texts(self, category):
        if category not in self.text_cache:
            phrases = ["a photo of a " + category]
            phrases.extend("a photo of a "+other for other in NAMES if other!=category)
            phrases.extend("a photo of "+other for other in BACKGROUND)
            packed = self.processor(text=phrases,return_tensors="pt",padding=True,truncation=True)
            features = self.model.get_text_features(**packed.to(self.device)).float()
            self.text_cache[category] = self.torch.nn.functional.normalize(features)
        return self.text_cache[category]

    def rank(self, rgb, rows, category):
        from PIL import Image
        image = np.asarray(rgb)
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
            raise ValueError("Expected uint8 RGB")
        category = noun(category)
        if not rows or not category:
            return []
        h,w = image.shape[:2]
        kept,crops = [],[]
        for row in rows:
            box = np.asarray(row.get("bbox",row.get("xyxy")),dtype=float)
            if box.shape!=(4,) or not np.all(np.isfinite(box)):
                raise ValueError("Malformed region")
            x,y,u,v = np.clip(box,[0,0,0,0],[w,h,w,h])
            if min(u-x,v-y)<=0:
                raise ValueError("Non-positive region")
            dx,dy = .12*(u-x),.12*(v-y)
            patch = image[max(0,int(y-dy)):min(h,int(v+dy)+1),max(0,int(x-dx)):min(w,int(u+dx)+1)]
            crops.append(Image.fromarray(patch))
            coords = [float(i) for i in (x,y,u,v)]
            kept.append(dict(row,bbox=coords,xyxy=coords))
        with self.torch.inference_mode():
            tf = self.texts(category)
            for offset in range(0,len(crops),12):
                packed = self.processor(images=crops[offset:offset+12],return_tensors="pt").to(self.device)
                features = self.model.get_image_features(**packed).float()
                sims = self.torch.nn.functional.normalize(features) @ tf.T
                scores = (sims[:,0]-sims[:,1:].max(dim=1).values).cpu().tolist()
                for row,score in zip(kept[offset:],scores):
                    row["review_margin"] = float(score)
        return sorted(kept,key=lambda r:-r["review_margin"])
