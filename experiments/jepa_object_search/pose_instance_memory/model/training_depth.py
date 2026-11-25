"""Replay the original training views; retain the unmodified validation/test depth file."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    p=argparse.ArgumentParser()
    for name in ("v20","assets","out"):
        p.add_argument(name,type=Path)
    p.add_argument("--house")
    a=p.parse_args()
    sys.path.insert(0,str(a.v20/"model"))
    from replay_depth import per_house
    manifest=a.v20/"data/manifest.jsonl"
    rows=[json.loads(l) for l in manifest.read_text().splitlines()]
    train=[(k,r) for k,r in enumerate(rows) if r["split"]=="train"]
    parts=a.out/"depth_parts"
    parts.mkdir(parents=True,exist_ok=True)
    a.manifest=manifest
    if a.house:
        per_house(a,[(i,r) for i,r in train if r["house"]==a.house])
        return
    houses=sorted({r["house"] for _,r in train})
    def work(h):
        with (parts/(h+".log")).open("w") as log:
            subprocess.run([sys.executable,__file__,str(a.v20),str(a.assets),str(a.out),"--house",h],
                           stdout=log,stderr=log,check=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(work,houses))
    fields={}
    for h in houses:
        with np.load(parts/(h+".npz")) as z:
            for k in z.files:
                fields.setdefault(k,[]).append(z[k])
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    with np.load(a.v20/"model/geometry.npz") as heldout:
        assert heldout["manifest_sha256"].item()==digest
        for k in fields:
            fields[k].append(heldout[k])
    fields={k:np.concatenate(values) for k,values in fields.items()}
    order=np.argsort(fields["ix"])
    assert np.array_equal(fields["ix"][order],np.arange(len(rows)))
    assert fields["valid"].all() and (fields["rgb_mae"]==0).all(), "RGB-depth correspondence not verified"
    np.savez_compressed(a.out/"geometry_all.npz",**{k:v[order] for k,v in fields.items()},manifest_sha256=digest)
    print(json.dumps({"training_views_replayed":len(train),"total_usable":len(order),"manifest_sha256":digest}))

if __name__=="__main__":
    main()
