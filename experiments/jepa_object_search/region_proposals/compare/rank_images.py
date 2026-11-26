"""Score proposals from frozen images without loading instance labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from region_rank import RegionRanker


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def index(records):
    result = {r["frame_id"]:r for r in records}
    if len(result)!=len(records):
        raise ValueError("Duplicate frame identifier")
    return result


def inputs(project, dataset, provider):
    run = project/"jepa_box_v26"
    hssd = dataset=="hssd"
    base = project/"jepa_reid_v21/data" if hssd else run/"habitat/private"
    manifest = rows(base/("frames.jsonl" if hssd else "manifest.jsonl"))
    images = [r for r in manifest if not hssd or r["split"]=="test"]
    path = {
        "sam":run/"sam/private/hssd/rows.jsonl" if hssd else base/"sam.jsonl",
        "dino":project/"jepa_reid_v21/detector/predictions.jsonl" if hssd else base/"dino.jsonl",
        "refined":run/"refine/private/refined.jsonl" if hssd else base/"refined.jsonl",
    }[provider]
    return base, images, path


def run(project,dataset,provider,device):
    base,images,det_path = inputs(project,dataset,provider)
    boxes = index(rows(det_path))
    out = project/"jepa_box_v26/compare/private"/f"{dataset}_{provider}.jsonl"
    digest = hashlib.sha256(Path(__file__).with_name("region_rank.py").read_bytes()).hexdigest()
    out.parent.mkdir(parents=True,exist_ok=True)
    if missing := [r for r in images if r["frame_id"] not in boxes]:
        raise RuntimeError(f"awaiting proposals for {len(missing)} frozen frames")
    model = RegionRanker(project,device)
    tmp = out.with_suffix(".working")
    with tmp.open("w") as stream:
        for frame in images:
            fid = frame["frame_id"]
            image = np.array(Image.open(base/frame["rgb"]).convert("RGB"))
            ranked = model.rank(image,boxes[fid]["boxes"],frame["category"])
            stream.write(json.dumps(dict(frame_id=fid,model_hash=digest,boxes=ranked),allow_nan=False)+"\n")
    tmp.replace(out)
    print(dataset,provider,len(images),"complete",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("project",type=Path)
    parser.add_argument("dataset",choices=["hssd","habitat"])
    parser.add_argument("provider",choices=["sam","dino","refined"])
    parser.add_argument("--device",default="cuda")
    a=parser.parse_args()
    run(a.project,a.dataset,a.provider,a.device)
