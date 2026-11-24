"""Compare a fresh physical replay to the original data, down to each RGB byte and instance ID."""
import json
from pathlib import Path
import sys
import numpy as np


def compare(root, name):
    original = root/"test"/name
    replay = root/"replay/test"/name
    files = {p.name for p in original.glob("*.npz")}
    assert len(files)==72 and files=={p.name for p in replay.glob("*.npz")}
    for file in sorted(files):
        with np.load(original/file) as a,np.load(replay/file) as b:
            assert set(a.files)==set(b.files)
            for field in a.files:
                if not np.array_equal(a[field],b[field]):
                    raise RuntimeError((file,field))
    result = dict(configurations=len(files),actual_routes=3*len(files),
        panorama_frames=12*len(files),future_frames=3*4*len(files),actions_out_and_back=(28+40)*3*len(files),
        repeated_observation_bytes_and_labels_equal=True)
    (root/"independent_replay.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":compare(Path(sys.argv[1]),sys.argv[2])
