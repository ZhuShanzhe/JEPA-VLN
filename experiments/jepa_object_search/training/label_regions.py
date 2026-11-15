"""Assign conservative HSSD semantic room presence to an immutable branch pack."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import tempfile

import numpy as np

from data_prep import Builder, read, records, require, sha, within


MARGIN_M = 0.25


def polygon_relation(point, region):
    """Return interior, boundary or outside in the horizontal region footprint."""
    if point[1] < region["floor_height"]-.16 or point[1] > region["floor_height"]+region["extrusion_height"]+.02:
        return "outside"
    xy = np.asarray(point)[[0,2]]
    vertices = np.array(region["poly_loop"])[:,[0,2]]
    following = np.roll(vertices,-1,axis=0)
    edges = following-vertices
    lengths = np.einsum("ij,ij->i",edges,edges)
    valid = lengths>1.e-10
    vertices, following, edges, lengths = vertices[valid], following[valid], edges[valid], lengths[valid]
    if len(vertices)<3:
        return "outside"
    diff = xy-vertices
    t = np.clip(np.einsum("ij,ij->i",diff,edges)/lengths,0,1)
    if np.linalg.norm(diff-t[:,None]*edges,axis=1).min()<MARGIN_M:
        return "boundary"
    crosses = (vertices[:,1]>xy[1]) != (following[:,1]>xy[1])
    if not crosses.any():
        return "outside"
    crossing_x = vertices[crosses,0] + (xy[1]-vertices[crosses,1])*edges[crosses,0]/edges[crosses,1]
    return "interior" if np.count_nonzero(crossing_x>xy[0])%2 else "outside"


def unique_region(point, regions):
    kinds = [polygon_relation(point,r) for r in regions]
    if kinds.count("interior")==1 and "boundary" not in kinds:
        return kinds.index("interior")
    return None


def read_scene(raw, scene, lexicon):
    regions=read(raw/"semantics/scenes"/f"{scene}.semantic_config.json")["region_annotations"]
    layout=read(raw/"scenes"/f"{scene}.scene_instance.json")
    stage=layout["stage_instance"]
    require(set(stage)=={"template_name"} and layout["translation_origin"]=="asset_local",
            "Scene transformation requires explicit coordinate validation")
    annotations=[{"present":set(), "uncertain":set(), "unknown":False} for _ in regions]
    asset_counts=Counter()
    for obj in layout["object_instances"]:
        asset=obj["template_name"]
        labels=lexicon.get(asset,set())
        for label in labels:
            asset_counts[label,asset]+=1
        kinds=[polygon_relation(obj["translation"],r) for r in regions]
        safe=kinds.count("interior")==1 and "boundary" not in kinds
        for room,kind in zip(annotations,kinds):
            if kind!="outside":
                room["unknown"] |= not bool(labels)
                room["present" if safe else "uncertain"].update(labels)
    return regions,annotations,asset_counts


def region_targets(root, row, xy, geometry, cache):
    scene, q=row["house"],row["query"]
    regions,annotations,asset_counts=geometry
    # Memory-swapped examples revisit source decisions. They retain the same regional truth.
    if row["id"].startswith("hssd/report-pair"):
        pair=row["id"].split("/")[1][len("report-pair"):]
        source=read(root/f"frontier_jepa/runs/hssd_report_train_pair{pair}/receipt.json")["source"]
    else:
        source=row["id"][len("hssd/"):]
    if source in cache:
        return cache[source]
    path=within(root/"a_exploration/paired_branches_v2",source)
    m=read(path/"manifest.json")
    require(m["source_scene"]==scene and m["frontier_count"]==len(xy),"HSSD source mismatch")
    teachers=read(path/"teacher/labels.json")
    labels=np.full(len(xy),-1,np.int8)
    expected=Counter(v["asset"] for v in teachers["instance_map"].values())
    if Counter({a:c for (name,a),c in asset_counts.items() if name==q})!=expected:
        cache[source]=labels
        return labels
    x,y,z,qx,qy,qz,qw=m["frozen_pose_xyz_xyzw"]
    require(np.max(np.abs([qx,qy,qz]))<.001 and abs(abs(qw)-1)<.001,"unexpected decision heading")
    home=unique_region((x,y,z),regions)
    if home is None:
        cache[source]=labels
        return labels
    for i,(forward,left) in enumerate(xy):
        with np.load(path/f"branch{i:03d}.npz") as f:
            end=f["relative_pose_xyyaw"][-1]
        prediction=unique_region((x-left,y,z-forward),regions)
        # A frontier on a doorway or before it does not license a label for the other side.
        if prediction is None or prediction==home or prediction!=unique_region((x-end[1],y,z-end[0]),regions):
            continue
        room=annotations[prediction]
        if q in room["present"]:
            labels[i]=1
        elif q not in room["uncertain"] and not room["unknown"]:
            labels[i]=0
    cache[source]=labels
    return labels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--base-pack",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    require(not args.output.exists(),"Output must be new")
    root=args.root.resolve()
    raw=root/"simulation/hssd"
    require(sha(args.base_pack/"index.jsonl")==read(args.base_pack/"receipt.json")["index_sha256"],
            "unverified starting pack")
    with (raw/"semantics/objects.csv").open() as f:
        lexicon={row["id"]:{s.strip() for k in ("wnsynsetkey","main_wnsynsetkey")
                            for s in row[k].split(",") if s.strip()} for row in csv.DictReader(f)}
    geometry, cache = {}, {}
    args.output.parent.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as t:
        work=Path(t)/"pack"
        (work/"states").mkdir(parents=True)
        b=Builder(root,work,"cpu")
        coverage=defaultdict(Counter)
        for row in records(args.base_pack/"index.jsonl"):
            path=within(args.base_pack,row["path"])
            require(sha(path)==row["sha256"],"changed base example")
            with np.load(path) as f:
                if row["source"]=="hssd":
                    house=row["house"]
                    if house not in geometry:
                        geometry[house]=read_scene(raw,house,lexicon)
                    presence=region_targets(root,row,f["xy"],geometry[house],cache)
                else:
                    presence=f["region_presence"]
                coverage[row["source"]+"/"+row["split"]].update(map(int,presence))
                b.add({k:v for k,v in row.items() if k not in ("sha256","path")},f["context"],f["future"],
                      f["xy"],f["actions"],f["gain"],float(f["step_m"]),float(f["turn_deg"]),
                      f["memory"] if f["has_memory"] else None,f["memory_xy"] if f["has_memory"] else None,
                      presence)
        b.finish()
        info=dict(base_index=sha(args.base_pack/"index.jsonl"), margin_m=MARGIN_M, coverage=coverage,
                  semantics="Goal-category instance translation in a uniquely annotated new room; nominal frontier and executed endpoint must agree. Boundary/overlap ambiguous; unmatched class inventory excluded; negatives also require classified objects within that room.")
        (work/"region_audit.json").write_text(json.dumps(info,indent=2)+"\n")
        print(json.dumps(info))
        work.rename(args.output)


if __name__=="__main__":
    main()
