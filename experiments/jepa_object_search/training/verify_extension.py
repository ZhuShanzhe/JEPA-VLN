"""Confirm an incremental pack kept all frozen data and newly encoded rows match their raw controls."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from data_prep import read, records, require, sha, within


def verify(base, raw, extension):
    before = records(base/"index.jsonl")
    added = records(raw/"index.jsonl")
    after = records(extension/"index.jsonl")
    for directory in (base,extension):
        require(read(directory/"receipt.json")["index_sha256"]==sha(directory/"index.jsonl"),
                "Frozen pack index changed")
    require(len(after)==len(before)+len(added) and len({r["id"] for r in after})==len(after),
            "Missing, duplicated or replaced rows")
    for row in after:
        require(sha(within(extension,row["path"]))==row["sha256"],"Extended state file changed")
    require(before==after[:len(before)],"Base index entries or member hashes changed")
    for r, packed in zip(added,after[len(before):]):
        require(packed["id"]=="extra/"+r["id"] and all(packed[k]==r[k] for k in
                ("house","split","query","label_kind")), "Raw/packed metadata mismatch")
        with np.load(within(raw,r["path"])) as x,np.load(within(extension,packed["path"])) as y:
            for k in ("xy","actions","gain","step_m","turn_deg","region_presence"):
                if k in x:
                    require(np.array_equal(x[k],y[k]),"Controls, labels or units changed")
    houses={}
    for r in after:
        require(houses.setdefault((r["source"],r["house"]),r["split"])==r["split"],
                "House shared between train and development")
    return dict(base_index_sha256=sha(base/"index.jsonl"),raw_index_sha256=sha(raw/"index.jsonl"),
                final_index_sha256=sha(extension/"index.jsonl"),preserved_base_rows=len(before),
                new_rows=len(added),by_split=dict(Counter(r["split"] for r in added)),
                total_rows=len(after),sample_hashes_verified=True,control_label_identity_verified=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("base","raw","extension"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    text=json.dumps(verify(a.base,a.raw,a.extension),indent=2)+"\n"
    if a.output:
        with a.output.open("x") as f:
            f.write(text)
    print(text,end="")
