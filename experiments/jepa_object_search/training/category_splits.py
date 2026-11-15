"""Plan class-held-out supervision folds without optimizing parameters or consulting scores."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


BASE = {"chair.n.01":"chair", "blue chair":"chair", "red chair":"chair",
        "sofa.n.01":"sofa", "sofa":"sofa", "cup":"cup", "bed.n.01":"bed",
        "toilet.n.02":"toilet", "television_receiver.n.01":"television",
        "potted_plant.n.01":"potted plant"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    raw=(args.data/"index.jsonl").read_bytes()
    rows=[json.loads(line) for line in raw.splitlines()]
    classes={BASE[r["query"]] for r in rows}
    dev_houses={(r["source"],r["house"]) for r in rows if r["split"]=="dev"}
    assert not dev_houses.intersection((r["source"],r["house"]) for r in rows if r["split"]=="train")
    folds={}
    for target in sorted(classes):
        train=[r for r in rows if r["split"]=="train" and BASE[r["query"]]!=target]
        dev=[r for r in rows if r["split"]=="dev" and BASE[r["query"]]==target]
        if not train or not dev:
            raise ValueError(f"Incomplete class fold for {target}")
        folds[target]=dict(train=[r["id"] for r in train],dev=[r["id"] for r in dev],
                           train_size=len(train),dev_size=len(dev),
                           dev_houses=len({(r["source"],r["house"]) for r in dev}))
    payload=dict(scope="Future development folds; held-out target supervision for the head. Frozen visual/language pretraining can contain every category.",
                 index_sha256=hashlib.sha256(raw).hexdigest(),folds=folds)
    with args.output.open("x",encoding="utf8") as stream:
        json.dump(payload,stream)
    print(json.dumps({k:{i:f[i] for i in ("train_size","dev_size","dev_houses")} for k,f in folds.items()}))


if __name__=="__main__":
    main()
