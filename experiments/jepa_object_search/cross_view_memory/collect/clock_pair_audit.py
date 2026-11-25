"""Aggregate same-clock evidence and confusion across different clock subtypes and templates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def confusion(labels, scores, threshold, selected):
    y=labels[selected].astype(bool)
    positive=(scores[selected]>=threshold)
    return dict(positive=int(y.sum()), negative=int((~y).sum()),
                tp=int((positive & y).sum()), fn=int((~positive & y).sum()),
                fp=int((positive & ~y).sum()),tn=int((~positive & ~y).sum()))


def audit(a):
    rows=[json.loads(line) for line in a.manifest.read_text().splitlines()]
    thresh=json.loads(a.metrics.read_text())["thresholds"]
    result={"clock_observations":len(rows),"clock_instances":len({r["instance"] for r in rows}),
            "houses":len({r["house"] for r in rows}),"subtype_images":Counter(r["subtype"] for r in rows),
            "manifest_sha256":hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
            "main_metrics_sha256":hashlib.sha256(a.metrics.read_bytes()).hexdigest(),"models":{}}
    with np.load(a.pairs) as f:
        i,j,y=f["i"],f["j"],f["y"]
        assert all(rows[x]["house"]==rows[z]["house"] and (rows[x]["instance"]==rows[z]["instance"])==bool(l) for x,z,l in zip(i,j,y))
        all_rows=np.ones(len(i),dtype=bool)
        same_template=np.asarray([rows[x]["asset"]==rows[z]["asset"] for x,z in zip(i,j)])
        same_subtype=np.asarray([rows[x]["subtype"]==rows[z]["subtype"] for x,z in zip(i,j)])
        result["distinct_physical_instance_pairs_by_negative_kind"]={}
        for label,mask in (("same_asset_template",same_template),
                           ("same_subtype_different_template",same_subtype&~same_template),
                           ("different_subtype",~same_subtype)):
            result["distinct_physical_instance_pairs_by_negative_kind"][label]=len({
                tuple(sorted((rows[x]["instance"],rows[z]["instance"])))
                for x,z in zip(i[(y==0)&mask],j[(y==0)&mask])})
        for model in ("frozen","direct","jepa"):
            scores=f["s_"+model]
            t=thresh[model]
            result["models"][model]={k:confusion(y,scores,t,mask) for k,mask in
                (("all",all_rows),("same_template_negative",(y==0)&same_template),
                 ("same_subtype_different_template_negative",(y==0)&same_subtype&~same_template),
                 ("different_subtype_negative",(y==0)&~same_subtype))}
            result["models"][model]["threshold"]=t
    a.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("manifest","pairs","metrics","out"):
        p.add_argument(k,type=Path)
    audit(p.parse_args())
