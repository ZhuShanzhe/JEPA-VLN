"""Export the two original heads with their frozen nearby-review thresholds."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_review(project):
    root = project / "jepa_reid_v21"
    out = root / "hard_cases"
    results = json.loads((out / "summary.json").read_text())
    verified = {}
    for name in ("pose_direct", "pose_jepa"):
        source = root / "model" / (name + ".pt")
        dest = out / (name + "_near_review.pt")
        source_sha = digest(source)
        base = torch.load(source, map_location="cpu", weights_only=True)
        threshold = results["methods"][name]["global_and_near_5pct"]["threshold"]
        review = {**base, "thresholds": {**base["thresholds"], "detected": threshold}}
        torch.save(review, dest)
        actual = torch.load(dest, map_location="cpu", weights_only=True)
        assert base.keys() == actual.keys() and base["state"].keys() == actual["state"].keys()
        assert all(torch.equal(tensor, actual["state"][key]) for key, tensor in base["state"].items())
        assert actual["thresholds"] == review["thresholds"] and digest(source) == source_sha
        verified[name] = {"source_sha256": source_sha, "review_sha256": digest(dest),
                          "identical_parameter_tensors": len(base["state"]),
                          "detected_threshold": threshold}
    (out / "review_exports.json").write_text(json.dumps(verified, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    export_review(parser.parse_args().project.resolve())
