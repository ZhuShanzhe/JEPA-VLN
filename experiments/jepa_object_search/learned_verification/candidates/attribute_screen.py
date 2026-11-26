"""Frozen CLIP crop versus a complete goal and visually confusable alternatives."""
import atexit
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np
from source_quality import coarse_category


def prompts(goal):
    category=coarse_category(goal)
    pos=[str(goal).strip()]
    neg=["an empty interior wall","an indoor door","a window with a view outdoors","an empty picture frame","a mirror reflecting a room"]
    if category=="painting" and "landscape" in goal.lower() and ("color" in goal.lower() or "colour" in goal.lower()):
        pos.append("a framed full-color landscape artwork depicting natural scenery")
        neg+=["a black and white landscape picture","a colorful abstract painting","a portrait painting of a person"]
    elif category=="clock":
        if len(goal.split())<9:
            pos.append("a clock showing the time")
        neg+=["a round dinner plate","a framed wall picture","a wall decoration without a clock"]
    return pos,neg


def local_crop(rgb,box):
    h,w=rgb.shape[:2]
    x,y,u,v=map(float,box)
    dx=(u-x)*.12; dy=(v-y)*.12
    return np.ascontiguousarray(rgb[max(0,int(y-dy)):min(h,int(v+dy+1)),max(0,int(x-dx)):min(w,int(u+dx+1))])


class AttributeWorker:
    def __init__(self,project,device):
        from proposal_worker import ProposalWorker
        self.proc=subprocess.Popen([str(project/"b_verification/env/bin/python"),"-u",__file__,str(project),device],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,env={**os.environ,"TRANSFORMERS_OFFLINE":"1","HF_HUB_OFFLINE":"1"})
        self.read=lambda sec:ProposalWorker.read(self,sec)
        self.close=lambda:ProposalWorker.close(self)
        atexit.register(self.close)
        try:
            if self.read(150)!={"ready":True}:raise RuntimeError("Attribute scorer failed to start")
        except Exception:
            self.close()
            raise

    def margin(self,rgb,goal):
        buf=io.BytesIO()
        np.savez(buf,rgb=rgb,goal=str(goal))
        msg=buf.getvalue()
        self.proc.stdin.write(struct.pack(">I",len(msg))+msg);self.proc.stdin.flush()
        result=self.read(60)
        if "error" in result:raise RuntimeError(result["error"])
        return float(result["margin"])


def serve(project,device):
    reply=sys.stdout;sys.stdout=sys.stderr
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import CLIPModel,CLIPProcessor
    ck=project/"comparisons_nov14/weights/openai--clip-vit-large-patch14"
    model=CLIPModel.from_pretrained(ck,local_files_only=True).eval().to(device)
    processor=CLIPProcessor.from_pretrained(ck,local_files_only=True)
    texts={}
    print(json.dumps({"ready":True}),file=reply,flush=True)
    stream=sys.stdin.buffer
    with torch.inference_mode():
        while header:=stream.read(4):
            n,=struct.unpack(">I",header)
            if n>64*1024*1024:raise ValueError("Image too large")
            blob=stream.read(n)
            if len(blob)!=n:raise EOFError()
            try:
                with np.load(io.BytesIO(blob),allow_pickle=False) as data:
                    goal=data["goal"].item(); image=Image.fromarray(data["rgb"])
                if goal not in texts:
                    pos,neg=prompts(goal)
                    token=processor(text=pos+neg,return_tensors="pt",padding=True,truncation=True).to(device)
                    vec=F.normalize(model.get_text_features(**token).float(),dim=-1)
                    if len(texts)>32:texts.clear()
                    texts[goal]=(vec,len(pos))
                token=processor(images=image,return_tensors="pt").to(device)
                v=F.normalize(model.get_image_features(**token).float(),dim=-1)
                vec,k=texts[goal]
                scores=(vec@v.T).flatten()
                answer={"margin":float(scores[:k].mean()-scores[k:].max())}
            except Exception as exc:
                answer={"error":type(exc).__name__}
            print(json.dumps(answer,allow_nan=False),file=reply,flush=True)


if __name__=="__main__":serve(Path(sys.argv[1]),sys.argv[2])
