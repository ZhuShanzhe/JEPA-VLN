"""Replay a causal, optional memory write using a bridge frame before the probe arrives."""
from collections import Counter,defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def run(project,output):
    sys.path[:0]=[str(project/n) for n in ("identity_stage3_model/train","jepa_reid_v21/association")]
    from evaluate import label_boxes
    from identity import IdentityVerifier
    root=project/"identity_stage3_audit/revisit/private"
    frames={r["name"]:r for r in read(root/"frames.jsonl")}
    boxes={r["name"]:r["boxes"] for r in read(root/"dino.jsonl")}
    labels={k:[t["instance"] if t else None for t in label_boxes(boxes[k],r["boxes"])] for k,r in frames.items()}
    groups=defaultdict(dict)
    for k,r in frames.items():
        groups[r["group"]][r["stage"]]=k
    models={m:IdentityVerifier(project,m) for m in ("direct","jepa")}
    @lru_cache(maxsize=None)
    def encoded(k):
        r=frames[k]
        with Image.open(root/"observations"/(k+".png")) as im:
            rgb=np.array(im.convert("RGB"))
        with np.load(root/"observations"/(k+".npz")) as data:
            return models["direct"].encode(rgb,boxes[k],data["depth"],np.array(r["K"]),np.array(r["T_world_from_camera"]))
    def match(model,ref,k):
        if ref is None or k is None:
            return None
        s,i=ref
        matrix=model.score(encoded(s),encoded(k))[i:i+1]
        decision,=model.decide(matrix)
        return decision["index"] if decision["state"]=="same" else None
    def score(k,i,uid):
        if i is None:
            return "withheld"
        label=labels[k][i]
        return "unverified" if label is None else "correct" if label==uid else "wrong"
    results={}
    for mode,model in models.items():
        transitions=defaultdict(Counter)
        arms=defaultdict(lambda:defaultdict(Counter))
        writes=Counter()
        for views in groups.values():
            sk=views["anchor"]
            uid=frames[sk]["uid"]
            src=(sk,labels[sk].index(uid)) if uid in labels[sk] else None
            bk=views.get("bridge")
            bi=match(model,src,bk)
            if bi is not None:
                bridge=(bk,bi)
                writes[score(bk,bi,uid)] += 1
            else:
                bridge=None
                writes["not_written"] += 1
            for qstage in ("same_side","opposite_side","closer"):
                if qstage not in views:
                    continue
                q=views[qstage]
                a,b=match(model,src,q),match(model,bridge,q)
                selected=a if b is None else b if a is None or a==b else None
                kind="visible" if any(t["instance"]==uid for t in frames[q]["boxes"]) else "not_annotated"
                old=score(q,a,uid)
                new=score(q,selected,uid)
                transitions[kind][old+"_to_"+new]+=1
                arms[kind]["anchor_only"][old]+=1
                arms[kind]["causal_second_reference"][new]+=1
        assert sum(writes.values())==36 and sum(transitions["visible"].values())==89
        results[mode]=dict(writes=writes,query_transitions=transitions,outcomes=arms)
    output.write_text(json.dumps(dict(results=results,paid_calls=0,
                    inputs_sha256=hashlib.sha256((root/"frames.jsonl").read_bytes()).hexdigest()),
                    indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    run(Path(sys.argv[1]),Path(sys.argv[2]))
