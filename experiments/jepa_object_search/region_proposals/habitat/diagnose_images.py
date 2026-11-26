"""Render private overlays and count target-pixel coverage on predeclared examples."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def rows(path):
    return {r["frame_id"]:r for r in (json.loads(line) for line in path.read_bytes().splitlines())}


def iou(a,b):
    a,b=np.asarray(a),np.asarray(b)
    inter=np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
    return float(inter / max(1e-9,np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter))


def profile(boxes,truth,mask,width=518):
    if not boxes:
        return dict(has_proposal=False,top_iou=0.,coverage=0.)
    box=boxes[0]["bbox"]
    gt=max(truth,key=lambda t:iou(t["xyxy"],box))
    x,y,u,v=np.clip(np.rint(np.asarray(box)*mask.shape[1]/width),0,mask.shape[1]).astype(int)
    overlap=(mask[y:v,x:u]==gt["instance"]).sum()
    return dict(has_proposal=True,top_iou=round(iou(gt["xyxy"],box),4),
                coverage=round(float(overlap/gt["pixels"]),4))


def chart(folder,frame,truth,named,caption,path):
    image=Image.open(folder/frame["rgb"]).convert("RGB")
    seg=np.asarray(Image.open(folder/truth["mask"]))
    valid=[t for t in truth["boxes"] if t["pixels"]>=103]
    fontfile="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    def font(n):
        return ImageFont.truetype(fontfile,n) if Path(fontfile).is_file() else ImageFont.load_default()
    edge=image.width
    header,footer=76,65
    canvas=Image.new("RGB",(2*edge,header+2*(edge+footer)),(247,247,244))
    draw=ImageDraw.Draw(canvas)
    draw.text((14,7),caption,fill="#10202d",font=font(25))
    contaminated=caption.startswith("Clock")
    key=("Green = full semantic-ID envelope (includes foreground ledge)   |   Magenta = predicted clock"
         if contaminated else "Green = semantic-object envelope   |   Magenta = top reviewed region")
    draw.text((14,41),key,fill="#334654",font=font(16))
    stats={}
    for index,(label,pred) in enumerate([("RGB + semantic-ID envelope",None)]+list(named.items())):
        crop=image.copy()
        pencil=ImageDraw.Draw(crop)
        for obj in valid:
            pencil.rectangle(obj["xyxy"],outline="#42ff72",width=4)
        if pred:
            pencil.rectangle(pred[0]["bbox"],outline="#fa39c3",width=3)
        ox=index%2*edge
        oy=header+index//2*(edge+footer)
        canvas.paste(crop,(ox,oy))
        draw.text((ox+10,oy+edge+4),label,fill="#142331",font=font(18))
        desc="Envelope includes the clock AND the separate foreground ledge." if contaminated else "Score = bounding-box overlap (IoU)"
        if pred is not None:
            s=stats[label]=profile(pred,valid,seg)
            if not s["has_proposal"]:
                desc="No candidate"
            elif contaminated:
                desc=f'Visually on the clock | IoU to full semantic envelope: {s["top_iou"]:.2f}'
            else:
                desc=(f'IoU {s["top_iou"]:.2f} | semantic pixels covered {100*s["coverage"]:.0f}% | '
                      +("hit" if s["top_iou"]>=.5 else "miss"))
        draw.text((ox+10,oy+edge+32),desc,fill="#3b4d58",font=font(15))
    canvas.save(path)
    return stats


if __name__=="__main__":
    root=Path(os.environ["NAV_ROOT"])/"jepa_box_v26"
    folder=root/"habitat/private"
    manifest,gt=rows(folder/"manifest.jsonl"),rows(folder/"gt.jsonl")
    heads={label:rows(root/"compare/private"/f"habitat_{name}.jsonl") for label,name in (
        ("Current Grounding DINO","dino"),("Original SAM proposals","sam"),("Trained DINO box refiner","refined"))}
    keys={k:next(fid for fid,row in manifest.items() if row["category"]==category and
                  row["group"]==group and predicate(fid)) for k,category,group,predicate in [
        ("clock","clock","archived",lambda f:f.endswith("after_approach")),
        ("painting","wall_art","target_facing",lambda f:any(b["pixels"]>=103 for b in gt[f]["boxes"]))]}
    summary={}
    for k,fid in keys.items():
        ss={label:pred[fid]["boxes"] for label,pred in heads.items()}
        path=folder/f"{k}_comparison.png"
        summary[k]=chart(folder,manifest[fid],gt[fid],ss,"Clock: historical close-up" if k=="clock" else
                         "Painting: first visible target-facing control",path)
    (root/"habitat/illustration_scores.json").write_text(json.dumps(summary,indent=2)+"\n")
