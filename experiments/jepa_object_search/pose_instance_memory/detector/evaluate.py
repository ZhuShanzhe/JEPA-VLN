"""Evaluate full-frame RGB proposals against eligible physical instances, one-to-one."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from proposals import LocalCandidateDetector,MODES,iou,select


def counts(frame, boxes):
    truth=frame["boxes"]
    overlap=np.array([[iou(b["xyxy"],g["xyxy"]) for g in truth] for b in boxes],dtype=float)
    matched=set()
    if boxes and truth:
        i,j=linear_sum_assignment(-((overlap>=.50)*1000+overlap))
        matched={index for index,column in zip(i,j) if overlap[index,column]>=.50}
    hits=len(matched)
    duplicate=sum(k not in matched and any(iou(b["xyxy"],g["xyxy"])>=.5 for g in truth)
                  for k,b in enumerate(boxes))
    return Counter(frames=1, truth=len(truth), proposed=len(boxes), true_positives=hits,
                   unmatched_proposals=len(boxes)-hits,duplicate_proposals=duplicate,
                   empty_target_frames=int(not truth),empty_target_frames_with_proposal=int(not truth and bool(boxes)),
                   multiple_frames=int(len(truth)>=2),multiple_with_two_recovered=int(len(truth)>=2 and hits>=2),
                   multiple_all_recovered=int(len(truth)>=2 and hits==len(truth)),
                   frames_with_targets=int(bool(truth)),positive_frames_with_a_match=int(hits>0))


def summarize(frames,rows,mode):
    by_cat=defaultdict(Counter)
    overall=Counter()
    raw_total=0
    for f in frames:
        raw=rows[f["frame_id"]]
        base=[b for b in raw if b["view"]=="full"]
        raw_total+= len(base) if mode=="base" or (mode=="empty" and base) else len(raw)
        c=counts(f,select(raw,mode))
        by_cat[f["category"]].update(c)
        overall.update(c)
    return dict(totals=dict(overall),raw_before_nms=raw_total,
                by_category={k:dict(c) for k,c in sorted(by_cat.items())})


def choose(tables):
    base=tables["base"]["totals"]
    by={}
    for name in MODES:
        t=tables[name]["totals"]
        if (t["unmatched_proposals"]-base["unmatched_proposals"] <= .15*t["frames"] and
            t["duplicate_proposals"]-base["duplicate_proposals"] <= .04*t["frames"]):
            by[name]=(t["true_positives"],t["multiple_with_two_recovered"],-t["unmatched_proposals"],-MODES.index(name))
    return max(by,key=by.get)


def atomic(p,value):
    temp=p.with_suffix(".tmp")
    temp.write_text(json.dumps(value,indent=2)+"\n")
    temp.replace(p)


def main(project,device):
    data=project/"jepa_reid_v21/data"
    dest=project/"jepa_reid_v21/detector"
    src=data/"frames.jsonl"
    frames=[json.loads(line) for line in src.read_text().splitlines() if line]
    assert len({r["frame_id"] for r in frames})==len(frames)
    model=LocalCandidateDetector(project,device,"augment")
    rows={}
    cache=dest/"rgb_cache"
    cache.mkdir(exist_ok=True)
    def infer(batch):
        for k,r in enumerate(batch):
            path=data/r["rgb"]
            key=hashlib.sha256((str(r["frame_id"])+r["category"]+device).encode()+path.read_bytes()+
                               (dest/"proposals.py").read_bytes()).hexdigest()
            cachepath=cache/(key+".json")
            if cachepath.exists():
                values=json.loads(cachepath.read_text())
            else:
                im=np.asarray(Image.open(path).convert("RGB"))
                # Only the query class and complete RGB image reach the model.
                values=model.proposals(im,r["category"],"augment")
                atomic(cachepath,values)
            rows[r["frame_id"]]=values
            if k%32==0:
                print("frame",k+1,"/",len(batch),flush=True)
    val=[r for r in frames if r["split"]=="val"]
    test=[r for r in frames if r["split"]=="test"]
    train=[r for r in frames if r["split"]=="train"]
    split_houses=[{r["house"] for r in group} for group in (train,val,test)]
    assert val and test and sum(map(len,split_houses))==len(set().union(*split_houses))
    start=time.monotonic()
    infer(val)
    calibration={m:summarize(val,rows,m) for m in MODES}
    chosen=choose(calibration)
    cfg=dict(selected_mode=chosen,box_threshold=.35,text_threshold=.25,nms_iou=.85,
             window_fraction=.80,cut_edge_exclusion_pixels=3,score_iou=.50,
             calibration_frames=len(val),device=device,manifest_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
             validation_rule="Maximize 1:1 matches; allow <=0.15 extra unmatched proposals/frame and <=0.04 extra duplicates/frame vs full. Tie-break dual coverage, fewer unmatched, then simpler mode.")
    atomic(dest/"protocol.json",cfg)
    print("selected",chosen,flush=True)
    infer(test)
    tables={m:summarize(test,rows,m) for m in MODES}
    infer(train)
    out=dest/"predictions.jsonl"
    tmp=out.with_suffix(".tmp")
    with tmp.open("w") as f:
        for row in frames:
            raw=rows[row["frame_id"]]
            base=[b for b in raw if b["view"]=="full"]
            admitted=base if chosen=="base" or (chosen=="empty" and base) else raw
            f.write(json.dumps({"frame_id":row["frame_id"],"boxes":select(raw,chosen),
                                "raw_boxes":admitted},separators=(",",":"))+"\n")
    tmp.replace(out)
    atomic(dest/"detector_summary.json",dict(calibration=calibration,test=tables,selected_mode=chosen,
           elapsed_s=round(time.monotonic()-start,2),paid_requests=0,protocol=cfg))
    print("complete",len(frames),chosen,tables[chosen]["totals"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("project",type=Path)
    p.add_argument("--device",default="cuda")
    a=p.parse_args()
    main(a.project.resolve(),a.device)
