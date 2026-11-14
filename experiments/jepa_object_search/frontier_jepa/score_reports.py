"""Run a one-room report-memory checkpoint on a decision-only JSON request."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from memory import load_memory_ranker
from vision import ContextEncoder


def validate_keys(record, allowed):
    if set(record) != set(allowed):
        raise ValueError(f"Inputs must contain exactly these fields: {sorted(allowed)}")


def picture(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def score_case(path, model, encoder):
    req = json.loads(path.read_text())
    validate_keys(req,["observed_rgb","candidates","reports"])
    if not req["observed_rgb"] or not req["candidates"]:
        raise ValueError("Expected a current observation and at least one candidate")
    current = np.stack([picture(path.parent / f) for f in req["observed_rgb"]])
    xy,plans,ids = [],[],[]
    for c in req["candidates"]:
        validate_keys(c,["id","relative_xz_m","planned_actions"])
        ids.append(c["id"])
        xy.append(c["relative_xz_m"])
        plans.append(c["planned_actions"])
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate identifiers must be unique")
    reports, poses = [],[]
    for r in req["reports"]:
        validate_keys(r,["observed_rgb","observed_camera_xz_m"])
        reports.append(encoder.encode(picture(path.parent / r["observed_rgb"])[None]))
        poses.append(r["observed_camera_xz_m"])
    scores = model.score(encoder.encode(current),xy,plans,np.array(reports).reshape(-1,768),
                         np.array(poses).reshape(-1,2))
    return {"request":path.name,"scores":dict(zip(ids,scores)),"preferred":ids[int(np.argmax(scores))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--requests",nargs="+",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new path")
    model = load_memory_ranker(args.checkpoint)
    encoder = ContextEncoder()
    results = [score_case(req,model,encoder) for req in args.requests]
    with args.output.open("x") as f:
        json.dump(results,f,indent=2)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
