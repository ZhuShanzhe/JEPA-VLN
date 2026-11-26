"""Evaluate ranked held-fixed boxes against a separate, sealed visual-only annotation."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

LABEL_HASH="60ef2d9e27a7227f1a61d68c3e2ae8bbe00288fc2f7f1af1b8be81ddbb90fe6e"

def overlap(a,b):
    a,b=np.array(a),np.array(b)
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return inter/max(1,(a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter)

def run(root):
    base=root/"refine_blind"
    ann=base/"private/labels_frozen.json"
    assert hashlib.sha256(ann.read_bytes()).hexdigest()==LABEL_HASH
    labels=json.loads(ann.read_bytes())["rows"]
    lookup=json.loads((base/"private/lookup.json").read_bytes())
    results={}
    for arm in ("sam","dino","refined"):
        filename=root/f"compare/private/habitat_{arm}.jsonl"
        rr={r["frame_id"]:r["boxes"] for r in map(json.loads,filename.read_bytes().splitlines())}
        assert set(lookup.values())<=set(rr)
        groups={}
        examples={}
        for code,a in labels.items():
            fid=lookup[code]
            pred=rr[fid]
            rating=a["rating"]
            if rating in ("uncertain","no_apparent_face"):
                g=groups.setdefault(rating,dict(frames=0,top_box_offered=0,top_passes_gate=0))
                g["frames"]+=1
                g["top_box_offered"]+=int(bool(pred))
                g["top_passes_gate"]+=int(bool(pred) and pred[0]["review_margin"]>0)
                if "archived" in fid:
                    examples[fid.split("archived_")[-1]]=dict(rating=rating)
                continue
            b,=a["clock_faces"]
            scores=[float(overlap(p["bbox"],b)) for p in pred]
            first=scores[0] if scores else 0.
            thresholds=(.30,.50,.75)
            for key in (rating,"all_identifiable"):
                g=groups.setdefault(key,dict(frames=0,top1_by_threshold=[0,0,0],top3_at_half=0,
                                              all_at_half=0,top1_iou_sum=0.,top1_gate_and_half=0))
                g["frames"]+=1;g["top1_iou_sum"]+=first
                for i,t in enumerate(thresholds):
                    g["top1_by_threshold"][i]+=int(first>=t)
                g["top3_at_half"]+=int(max(scores[:3],default=0.)>=.5)
                g["all_at_half"]+=int(max(scores,default=0.)>=.5)
                g["top1_gate_and_half"]+=int(first>=.5 and pred[0]["review_margin"]>0)
            if "archived" in fid:
                examples[fid.split("archived_")[-1]]=dict(rating=rating,top_iou=round(first,4),best=round(max(scores,default=0.),4))
        for g in groups.values():
            if "top1_iou_sum" in g:
                g["mean_top1_iou"]=round(g.pop("top1_iou_sum")/g["frames"],4)
        results[arm]=dict(groups=groups,archived=examples)
    report=dict(protocol="Single AI visual inspection, 21 anonymized randomized 518x518 RGB images only. Visible extent of dials/numerals/hands; uncertain frames excluded rather than treated as background. Diagnostic for the visibly identifiable clock class, not official instance identity.",
                sealed_sha256=LABEL_HASH, top1_iou_thresholds=[.30,.50,.75], paid_calls=0,arms=results)
    (base/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:(v["groups"]["all_identifiable"],v["archived"]) for k,v in results.items()}))

if __name__=="__main__":
    run(Path(sys.argv[1]))
