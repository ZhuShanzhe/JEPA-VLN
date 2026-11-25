"""Verify cross-view and identity conditions and report template overlap without image disclosure."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path


def audit(root):
    records=[json.loads(l) for l in (root/"manifest.jsonl").read_text().splitlines()]
    ids=defaultdict(list)
    names=defaultdict(set)
    grouped=defaultdict(set)
    for r in records:
        ids[r["instance"]].append(r)
        names[r["split"]].add(r["asset"])
        grouped[r["split"],r["house"],r["category"]].add(r["instance"])
    category_counts={};camera_pairs=defaultdict(Counter)
    for views in ids.values():
        first=views[0]
        assert all(r["house"]==first["house"] and r["asset"]==first["asset"] and r["split"]==first["split"] and r["category"]==first["category"] for r in views)
        for a,b in itertools.combinations(views,2):
            angle=abs((a["yaw_deg"]-b["yaw_deg"]+180)%360-180)
            distance=math.dist(a["camera_xyz"],b["camera_xyz"])
            assert angle>=15-1e-6 and distance>=.35-1e-6
            delta=abs((a["camera_heading_deg"]-b["camera_heading_deg"]+180)%360-180)
            camera_pairs[first["split"]]["pairs"]+=1
            camera_pairs[first["split"]]["object_angle_at_least_45"]+=angle>=45
            camera_pairs[first["split"]]["camera_heading_at_least_45"]+=delta>=45
    result={}
    for split in ("train","val","test"):
        same_asset=different_asset=0
        by_category=defaultdict(Counter)
        for (s,h,c),group in grouped.items():
            if s!=split:
                continue
            by_category[c]["house_count"]+=1
            by_category[c]["instance_count"]+=len(group)
            for a,b in itertools.combinations(group,2):
                if ids[a][0]["asset"]==ids[b][0]["asset"]:
                    same_asset+=1
                else:
                    different_asset+=1
        subset=[r for r in records if r["split"]==split]
        result[split]=dict(
            houses=len({r["house"] for r in subset}), instances=len({r["instance"] for r in subset}), views=len(subset),
            asset_templates=len(names[split]), distinct_templates_absent_from_train=len(names[split]-names["train"]),
            views_with_template_absent_from_train=sum(r["asset"] not in names["train"] for r in subset),
            same_house_category_instance_pairs_same_asset=same_asset,
            same_house_category_instance_pairs_different_asset=different_asset,
            view_geometry=dict(camera_pairs[split]),category_counts=by_category)
    result["manifest_sha256"]=hashlib.sha256((root/"manifest.jsonl").read_bytes()).hexdigest()
    tmp=root/"geometry_audit.tmp"
    tmp.write_text(json.dumps(result,indent=2));tmp.replace(root/"geometry_audit.json")
    print(json.dumps({k:{x:y for x,y in v.items() if x!="category_counts"} for k,v in result.items() if isinstance(v,dict)}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    audit(p.parse_args().root)
