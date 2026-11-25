"""Load the real weights through the evaluator's optional inference bridge."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import cv2

from model_worker import MatcherProcess


def check(project):
    data = project / "jepa_reid_v20/data"
    weights = project / "jepa_reid_v20/model/jepa.pt"
    by_first = []
    for row in map(json.loads,(data / "manifest.jsonl").read_bytes().splitlines()):
        if row["split"] == "test" and (not by_first or row["instance"] == by_first[0]["instance"]):
            by_first.append(row)
            if len(by_first)==2:
                break
    assert len(by_first)==2 and by_first[0]["view"]!=by_first[1]["view"]
    def regions(row):
        def rgb(name):
            return cv2.cvtColor(cv2.imread(str(data / row[name])), cv2.COLOR_BGR2RGB)
        return rgb("image"),rgb("context_image"),row["box_in_context"]
    interpreter = project.parent / "jepa_instance_bench_20261016/policy_env/bin/python"
    worker = MatcherProcess(interpreter,project,weights,"cpu")
    try:
        a,b=by_first
        yaw=math.remainder(b["camera_heading_deg"]-a["camera_heading_deg"],360)
        result=worker.compare(regions(a),regions(b),yaw)
        assert 0 <= result["score"] <= 1 and result["match"] == (result["score"]>=result["threshold"])
    finally:
        worker.close()
    return {"purpose":"Two-environment integration smoke, one predetermined held-out image pair.",
            "evaluator_python":str(Path(sys.executable).relative_to(project)),
            "worker_python":str(interpreter.relative_to(project.parent)),
            "weight_sha256":hashlib.sha256(weights.read_bytes()).hexdigest(), "passed":True,
            "paid_requests":0, "response_fields":sorted(result)}


if __name__ == "__main__":
    print(json.dumps(check(Path(sys.argv[1])),indent=2))
