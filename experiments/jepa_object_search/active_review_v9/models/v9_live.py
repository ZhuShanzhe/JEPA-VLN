"""Online active-look policy. Inputs are previously observed RGB and past reported crops."""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from v9_core import Local, ReviewHead, descriptors, decide


class Features:
    def __init__(self, model_dir, device=None):
        import open_clip
        p=Path(model_dir)
        files=json.loads((p/"encoder.json").read_text())
        rows=torch.load(p/files["projection"],weights_only=True,map_location="cpu")
        self.names=rows["words"]
        net,_,self.prep=open_clip.create_model_and_transforms("RN50",pretrained=str(p/files["rn50"]),device="cpu")
        self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.visual=net.visual.to(self.device).eval()
        self.center=rows["center"].to(self.device)
        self.basis=rows["basis"].to(self.device)
        self.anchors=rows["anchors"].to(self.device)

    @torch.inference_mode()
    def encode(self,frames):
        batch=torch.stack([self.prep(Image.fromarray(np.ascontiguousarray(x[:,:,:3])).convert("RGB")) for x in frames]).to(self.device)
        pool=self.visual.attnpool
        cells=[]
        handle=pool.register_forward_pre_hook(lambda _,args:cells.append(args[0]))
        try:self.visual(batch)
        finally:handle.remove()
        local=F.normalize(pool.c_proj(pool.v_proj(cells[0].flatten(2).transpose(1,2))).float(),dim=-1)
        return ((local-self.center)@self.basis).half().cpu(), (local@self.anchors.T).half().cpu()


class Assistant:
    def __init__(self, model_dir):
        import json
        self.visual=Features(model_dir,os.environ.get("V9_ENCODER_DEVICE","cpu"))
        data=torch.load(Path(model_dir)/"v9.pt",map_location="cpu",weights_only=True)
        self.words=data["words"]
        assert list(self.visual.names)==self.words
        self.local=Local(len(self.words)).eval()
        self.head=ReviewHead().eval()
        self.local.load_state_dict(data["local"])
        self.head.load_state_dict(data["head"])
        self.cut=json.loads((Path(model_dir)/"policy.json").read_text())["threshold"]

    def match_query(self,text):
        text=text.lower().replace("_"," ")
        queries={q.split(".")[0].replace("_"," "):q for q in self.words}
        for word,q in sorted(queries.items(),key=lambda a:-len(a[0])):
            if word in text:return q
        mapping={"painting":"wall_art","picture":"picture_frame","wall clock":"clock","light":"lamp","plant":"potted_plant","tv":"television_receiver","couch":"sofa"}
        for word,canonical in mapping.items():
            if word in text:
                return next((q for q in self.words if canonical==q.split(".")[0]),None)
        return None

    @torch.inference_mode()
    def score(self,frames,query,headings,report_images=()):
        q=self.match_query(query)
        if q is None or not len(frames):
            return dict(action="continue",current_score=0.,turn_score=0.,thresholds={"review":self.cut},supported=False)
        if len(headings)!=len(frames) or headings[-1]!=0:raise ValueError("expected current relative bin zero")
        delta=0
        if len(frames)>1:
            value=(int(headings[-2])+6)%12-6
            if abs(value)==1:delta=value
        use=frames[-2:] if delta else frames[-1:]
        code,sim=self.visual.encode(use)
        code,sim=code.float(),sim.float()
        v,n=self.local(code,sim)
        memory=None
        if len(report_images):
            c,_=self.visual.encode(report_images[-1:])
            memory=c.float().reshape(7,7,-1)[2:5,2:5].mean((0,1))
        j=self.words.index(q)
        x=descriptors(v,n,sim,code,memory)[:,j]
        scores=self.head(x[-1],x[0],delta,float(memory is not None)).sigmoid().numpy()
        return dict(action=decide(scores,self.cut),current_score=float(scores[0]),turn_score=float(max(scores[1:])),
            left_score=float(scores[1]),right_score=float(scores[2]),thresholds={"review":self.cut},supported=True,canonical=q)
