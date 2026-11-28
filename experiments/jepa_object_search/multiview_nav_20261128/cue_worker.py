"""Persistent server-local detector; stdin/out contain only frame paths and boxes."""
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "jepa_reid_v21/detector"))


def main():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2")
    with redirect_stdout(sys.stderr):
        from proposals import LocalCandidateDetector, select
        cfg = json.loads((ROOT / "jepa_reid_v21/detector/protocol.json").read_text())
        assert (cfg["device"],cfg["box_threshold"],cfg["text_threshold"],cfg["nms_iou"],cfg["selected_mode"]) == ("cuda",.35,.25,.85,"augment")
        detector = LocalCandidateDetector(ROOT, device=cfg["device"], mode=cfg["selected_mode"])
    print(json.dumps({"status":"ready"}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        assert len(request["frames"]) == 3
        result = []
        with redirect_stdout(sys.stderr):
            for path in request["frames"]:
                with Image.open(path) as im:
                    rgb = np.asarray(im.convert("RGB"))
                assert rgb.ndim==3 and rgb.shape[2]==3 and min(rgb.shape[:2])>=2
                preds = select(detector.proposals(rgb,request["goal"]), "augment")
                result.append([{"xyxy":row["xyxy"]} for row in preds])
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
