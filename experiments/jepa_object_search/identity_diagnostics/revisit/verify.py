"""Check persisted observations against instance pixels and compare views on paired objects."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def run(root, project, out):
    frames = rows(root/"frames.jsonl")
    outcomes = rows(root/"outcomes.jsonl")
    assert len({r["name"] for r in frames}) == len(frames)
    seen = defaultdict(list)
    nboxes = 0
    for row in frames:
        rel = root/"observations"/row["name"]
        with Image.open(rel.with_suffix(".png")) as im:
            assert im.size == (384,384) and im.mode == "RGB"
            im.load()
        with np.load(rel.with_suffix(".npz"),allow_pickle=False) as f:
            depth, sem = f["depth"], f["semantic"]
            assert depth.shape == sem.shape == (384,384) and np.isfinite(depth).all() and (depth >= 0).all()
            assert sem.dtype == np.int32 and depth.dtype == np.float32
            for box in row["boxes"]:
                ys,xs = np.nonzero(sem == int(box["instance"].split(":")[-1]))
                assert len(ys) == box["pixels"] and len(ys) >= 170
                assert list(map(int,[xs.min(),ys.min(),xs.max()+1,ys.max()+1])) == box["xyxy"]
                nboxes += 1
        T = np.array(row["T_world_from_camera"])
        assert np.allclose(T[:3,:3].T @ T[:3,:3],np.eye(3),atol=1e-4) and np.linalg.det(T[:3,:3])>0
        assert all(np.linalg.norm(T[:3,3]-p)>=.0999 for p in seen[row["group"]])
        seen[row["group"]].append(T[:3,3])
    oldsplit = json.loads((project/"jepa_reid_v21/data/frozen_protocol.json").read_text())["splits"]
    current = {r["house"] for r in frames}
    assert current <= set(oldsplit["test"]) and current.isdisjoint(oldsplit["train"]+oldsplit["val"])
    groups = defaultdict(dict)
    for r in outcomes:
        if r["mode"]=="dino" and r["visible"]:
            groups[r["group"]][r["stage"]] = r["outcomes"]["first"]
    compare = {}
    for other in ("same_side", "opposite_side"):
        counts=Counter()
        for views in groups.values():
            if other in views and "closer" in views:
                close, side = views["closer"]=="correct", views[other]=="correct"
                counts["both_correct" if close and side else "closer_only" if close else "side_only" if side else "neither"]+=1
        compare[other]=dict(n=sum(counts.values()),**counts)
    reference_files = ["jepa_reid_v21/data/build_frames.py","jepa_reid_v20/model/replay_depth.py",
                       "jepa_reid_v21/detector/protocol.json","jepa_reid_v21/detector/proposals.py",
                       "jepa_report_v19/appearance.py","jepa_reid_v23/report/report_evidence.py",
                       "jepa_reid_v21/association/evaluate.py","jepa_reid_v21/association/geometry_links.py"]
    stats=dict(verified_images=len(frames),verified_depth_and_instance_maps=len(frames),
               verified_box_annotations=nboxes,physical_instances=len(seen),verified_heldout_project_houses=len(current),
               paired_single_reference_dino_closer_vs_side=compare,paid_calls=0,
               module_sha256={n:hashlib.sha256((project/n).read_bytes()).hexdigest() for n in reference_files})
    out.write_text(json.dumps(stats,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for field in ("data","project","output"):
        ap.add_argument(field,type=Path)
    args=ap.parse_args()
    run(args.data,args.project,args.output)
