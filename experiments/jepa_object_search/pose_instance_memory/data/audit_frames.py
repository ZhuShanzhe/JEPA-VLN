"""Audit source coverage, projection conventions and pair difficulty."""
from collections import Counter,defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def audit(root):
    path=root/"frames.jsonl"
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    splits=defaultdict(list); families={}
    for r in rows:
        splits[r["split"]].append(r)
        assert families.setdefault(r["house"].split("_")[0],r["split"])==r["split"]
    result={}
    for split,frames in splits.items():
        seq=defaultdict(list);stats=Counter();strata=Counter();cats=Counter()
        near_pairs=set();identical_pairs=set()
        poses=set()
        for r in frames:
            seq[r["sequence"]].append(r); cats[r["category"]]+=1
            p=np.array(r["T_world_from_camera"]);k=np.array(r["K"])
            assert p.shape==(4,4) and np.max(abs(p[:3,:3].T@p[:3,:3]-np.eye(3)))<1e-5 and np.linalg.det(p[:3,:3])>.9999
            pose=(r["house"],*np.round(p.flatten(),4));stats["duplicate_camera_poses"]+=pose in poses;poses.add(pose)
            with Image.open(root/r["rgb"]) as im:
                assert im.size==(384,384);im.verify()
            depth=np.load(root/r["depth"],allow_pickle=False);assert depth.shape==(384,384) and depth.dtype==np.float32
            valid=np.where(np.isfinite(depth)&(depth>0))
            assert len(valid[0])
            yy,xx=int(valid[0][len(valid[0])//2]),int(valid[1][len(valid[1])//2])
            z=float(depth[yy,xx])
            pix=np.array([xx+.5,yy+.5,1.])
            world=p[:3,:3]@(np.linalg.solve(k,pix)*z)+p[:3,3]
            back=k@(p[:3,:3].T@(world-p[:3,3]))
            assert np.max(abs(back[:2]/back[2]-pix[:2]))<.001
            boxes=r["boxes"];stats["gt_targets"]+=len(boxes);stats["zero_target_frames"]+=not boxes
            stats["single_target_frames"]+=len(boxes)==1; stats["multiple_target_frames"]+=len(boxes)>1
            stats["source_rgb_matches"]+=r["source_rgb_checked"]
            assert len({b["instance"] for b in boxes})==len(boxes)
            for a,b in combinations(boxes,2):
                key=tuple(sorted([a["instance"],b["instance"]]))
                stats["covisible_pairs"]+=1
                if math.dist(a["world_center"],b["world_center"])<=1.25:
                    near_pairs.add(key);stats["near_covisible_pairs"]+=1
                    if a["asset"]==b["asset"]:
                        identical_pairs.add(key);stats["near_covisible_same_asset"]+=1
        for items in seq.values():
            assert len(items)==3 and sorted(r["view"] for r in items)==[0,1,2]
            strata[items[0]["stratum"]]+=1
            anchor=next(r for r in items if r["view"]==0)
            if anchor["stratum"]=="single":assert len(anchor["boxes"])==1
            if anchor["stratum"]=="multiple":assert len(anchor["boxes"])>=2
            for a,b in combinations(items,2):
                first={n["instance"] for n in a["boxes"]};second={n["instance"] for n in b["boxes"]}
                stats["cross_view_same_instance_pairs"]+=len(first&second)
                stats["cross_view_different_instance_pairs"]+=len(first)*len(second)-len(first&second)
                stats["neighbor_pairs_each_reveals_exclusive_instance"]+=bool(first-second and second-first)
        result[split]=dict(houses=len({r["house"] for r in frames}),sequences=len(seq),frames=len(frames),
                           categories=dict(cats),strata=dict(strata),counts=dict(stats),
                           unique_near_distinct_pairs=len(near_pairs),unique_near_same_asset_pairs=len(identical_pairs))
    report=dict(splits=result,manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (root/"data_audit.json").write_text(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    print(audit(Path(sys.argv[1])))
