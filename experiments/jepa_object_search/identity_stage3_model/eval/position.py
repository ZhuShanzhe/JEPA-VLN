"""Position-only comparator; use measured surface points and the preexisting V20 threshold."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np

from compare import build, digest, sha, summarize


THRESHOLD_M=.347302806383428


def run(project,out):
    sys.path[:0]=[str(project/"jepa_reid_v21/model"),str(project/"jepa_reid_v21/association")]
    from encode_proposals import support
    from evaluate import label_boxes
    frames,tasks,protocol=build(project)
    assert protocol==json.loads((out/"protocol.json").read_text())
    @lru_cache(maxsize=None)
    def calibrated(k):
        fr=frames[k]
        d=fr["depth"]
        if d.suffix==".npz":
            with np.load(d,allow_pickle=False) as z:
                depth=z["depth"]
        else:
            depth=np.load(d,allow_pickle=False)
        boxes=fr["boxes"]
        labels=[None if x is None else x["instance"] for x in label_boxes(boxes,fr["truth"])]
        points=[support(depth,b,np.asarray(fr["K"]),np.asarray(fr["c2w"])) for b in boxes]
        return labels,points
    rec=[]
    for i,t in enumerate(tasks):
        sl,sp=calibrated(t["source"])
        cl,cp=calibrated(t["live"])
        pred=[]
        for uid in t["uids"]:
            state,outcome="uncertain","uncertain"
            src=sp[sl.index(uid)] if uid in sl else None
            if src is not None and len(cp)>0:
                distances=[float(np.linalg.norm(src-p)) if p is not None else np.inf for p in cp]
                j=int(np.argmin(distances))
                if distances[j]<=THRESHOLD_M:
                    state="same"
                    outcome="unverified" if cl[j] is None else "original" if cl[j]==t["uids"][0] else "other_instance"
            pred.append(dict(state=state,outcome=outcome,source_box=uid in sl,usable_current=uid in cl))
        rec.append(dict(case=i,panel=t["panel"],category=t["category"],house=t["house"],unit=t["unit"],view=t["view"],visible=t["visible"],pred=pred))
    p=out/"private"/"position.jsonl"
    p.parent.mkdir(exist_ok=True)
    p.write_text("".join(json.dumps(r)+"\n" for r in rec))
    agg=dict(method="Closest RGB-D box surface point, within a fixed radius; all other cases uncertain",
             threshold_m=THRESHOLD_M,point_implementation="jepa_reid_v21.model.encode_proposals.support",
             frozen_code_sha256=sha(project/"jepa_reid_v21/model/encode_proposals.py"),
             cases_sha256=digest(tasks),paid_calls=0,metrics=summarize(rec))
    (out/"position.json").write_text(json.dumps(agg,sort_keys=True,indent=2)+"\n")
    print(agg["metrics"]["visible"],agg["metrics"]["twins"])


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("project",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    run(a.project,a.output)
