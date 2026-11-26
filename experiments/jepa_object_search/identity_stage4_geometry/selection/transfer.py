"""Controlled HSSD test with exact *past* keyframe maps and RGB-only query localization."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image


def lines(path):
    return [json.loads(s) for s in path.read_bytes().splitlines()]


def source_view(bundle, K, pose, stride=8):
    depth = bundle[::stride,::stride]
    y,x = np.indices(depth.shape)
    pixels = np.stack([x*stride+.5,y*stride+.5,np.ones_like(x)],axis=-1)
    points = (pixels@np.linalg.inv(K).T)*depth[:,:,None]
    world = points@pose[:3,:3].T+pose[:3,3]
    return {"grid":world, "K":K, "c2w":pose, "stride":stride}


def summarize(items):
    yes = [r for r in items if r["reason"]=="localized"]
    def median(field):
        return round(float(np.median([r[field] for r in yes])),4) if yes else None
    return dict(queries=len(items), houses=len({r["house"] for r in items}),
        pose_available=len(yes), localized_over_25cm_or_10deg=sum(r["translation_m"]>.25 or r["rotation_deg"]>10 for r in yes),
        median_translation_error_m=median("translation_m"),median_rotation_error_deg=median("rotation_deg"),
        with_detector_candidates=sum(r["has_boxes"] for r in items),
        all_boxes_with_9_points=sum(r.get("all_boxes_supported",False) for r in items),
        median_cpu_ms=round(float(np.median([r["ms"] for r in items])),1) if items else None,
        reasons=dict(Counter(r["reason"] for r in items)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project",type=Path)
    ap.add_argument("output",type=Path)
    ap.add_argument("--near",type=Path)
    args=ap.parse_args()
    p=args.project.resolve()
    private=p/"identity_stage3_model/new_houses/private"
    provider=p/"identity_stage4_geometry/provider"
    sys.path.insert(0,str(provider))
    from anchored_tracking import estimate_from_anchor
    frames={r["name"]:r for r in lines(private/"frames.jsonl")}
    dino={r["name"]:r["boxes"] for r in lines(private/"dino.jsonl")}
    cases=[r for r in lines(private/"cases.jsonl") if r["kind"]=="same"]
    assert len(cases)==92
    cv2.setNumThreads(2)
    cv2.setRNGSeed(173)
    @lru_cache(maxsize=20)
    def rgb(name):
        with Image.open(private/"observations"/(name+".png")) as im:
            return np.asarray(im.convert("RGB"))
    @lru_cache(maxsize=20)
    def historical(name):
        with np.load(private/"observations"/(name+".npz"),allow_pickle=False) as z:
            depth=z["depth"]
        return rgb(name),source_view(depth,np.asarray(frames[name]["K"]),np.asarray(frames[name]["T"]))
    groups=defaultdict(list)
    for task in cases:
        s,q=task["source"],task["query"]
        if args.near:
            with np.load(args.near/(q+".npz"),allow_pickle=False) as z:
                anchor_rgb=z["rgb"]
                view=source_view(z["depth"],np.asarray(frames[q]["K"]),z["T"])
        else:
            anchor_rgb, view=historical(s)
        start=time.perf_counter()
        geom,diag=estimate_from_anchor(anchor_rgb,view,rgb(q))
        ms=(time.perf_counter()-start)*1000
        # Truth for the query is opened only after the RGB-only estimate returns.
        qt=np.asarray(frames[q]["T"])
        distance=float(np.linalg.norm(view["c2w"][:3,3]-qt[:3,3]))
        angle=np.rad2deg(np.arccos(np.clip((np.trace(view["c2w"][:3,:3].T@qt[:3,:3])-1)/2,-1,1)))
        record=dict(house=task["house"],reason=diag["reason"],ms=ms,has_boxes=bool(dino[q]))
        if geom is not None:
            depth,_,est=geom
            rotation=np.trace(qt[:3,:3].T@est[:3,:3])
            record["rotation_deg"]=float(np.rad2deg(np.arccos(np.clip((rotation-1)/2,-1,1))))
            record["translation_m"]=float(np.linalg.norm(est[:3,3]-qt[:3,3]))
            counts=[]
            for a,b,c,d in dino[q]:
                counts.append(int(np.isfinite(depth[max(0,int(b)):int(np.ceil(d)),max(0,int(a)):int(np.ceil(c))]).sum()))
            record["all_boxes_supported"]=bool(counts) and min(counts)>=9
        groups["all"].append(record)
        groups["within_1m_30deg" if distance<=1 and angle<=30 else "outside_1m_or_30deg"].append(record)
    result=dict(anchor="controlled preceding view" if args.near else "first archived instance view",
        oracle_input="depth + metric pose of past anchor only",query_input="RGB only",
        scope="sparse pose and detector-region availability; excludes picking the right instance and whole navigation",
        cases_sha256=hashlib.sha256((private/"cases.jsonl").read_bytes()).hexdigest(),
        tracker_sha256=hashlib.sha256((provider/"anchored_tracking.py").read_bytes()).hexdigest(),
        results={k:summarize(v) for k,v in groups.items()})
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(result, flush=True)


if __name__=="__main__":
    main()
