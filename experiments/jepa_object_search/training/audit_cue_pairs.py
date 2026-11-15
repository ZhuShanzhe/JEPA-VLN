"""Check whether swapped outcomes have distinct start RGB and distinct frozen visual inputs."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from audit_observability import summarize
from data_prep import read, records, require, sha, within


def audit(pack, raw_directories):
    receipt = read(pack / "receipt.json")
    require(sha(pack / "index.jsonl") == receipt["index_sha256"], "Grid checksum changed")
    table = {r["id"]: r for r in records(pack / "index.jsonl")}
    result = {"grid_index_sha256": receipt["index_sha256"], "sources": {}}
    for directory in raw_directories:
        stats = summarize(directory)
        data = records(directory / "index.jsonl")
        kinds = {r["label_kind"] for r in data}
        require(len(kinds)==1,"A single capture condition is required per raw directory")
        items = defaultdict(list)
        for r in data:
            row = table["extra/"+r["id"]]
            path = within(directory, r["path"])
            require(sha(path) == receipt["raw_blender_sha256"][r["id"]], "Raw capture changed")
            cached = within(pack,row["path"])
            require(sha(cached) == row["sha256"], "Frozen visual inputs changed")
            with np.load(path,allow_pickle=False) as f, np.load(cached,allow_pickle=False) as z:
                require(np.array_equal(z["gain"], f["gain"]),"Teacher changed")
                items[r["split"],r["house"],r["query"]].append((
                    f["context_rgb"], z["context"], f["gain"], f["xy"], f["actions"]))
        grouped = defaultdict(Counter)
        for (split,_,_), pair in items.items():
            require(len(pair)==2, "Missing paired counterbalanced layout")
            (img0,feature0,gain0,xy0,ac0),(img1,feature1,gain1,xy1,ac1) = pair
            require(np.array_equal(xy0,xy1) and np.array_equal(ac0,ac1),"Controls differ")
            distinct = int(np.any(img0 != img1,axis=-1).sum())
            feat = not np.array_equal(feature0,feature1)
            top0,top1 = gain0==gain0.max(),gain1==gain1.max()
            opposite = not (top0 & top1).any()
            cnt = grouped[split]
            cnt["paired_variants"]+=1
            cnt["opposite_optimal_direction_pairs"]+=int(opposite)
            cnt["distinct_visible_pixels_pairs"]+=int(distinct>0)
            cnt["distinct_model_input_grid_pairs"]+=int(feat)
            cnt["opposite_outcome_same_observation_pairs"]+=int(opposite and distinct==0)
            cnt["opposite_outcome_distinct_RGB_and_grid_pairs"]+=int(opposite and distinct>0 and feat)
            if "minimum_changed_RGB_pixels" not in cnt or distinct < cnt["minimum_changed_RGB_pixels"]:
                cnt["minimum_changed_RGB_pixels"]=distinct
        stats["matched_pairs"]={k:dict(v) for k,v in grouped.items()}
        result["sources"][next(iter(kinds))]=stats
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid",type=Path,required=True)
    parser.add_argument("--raw",action="append",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.grid,args.raw)
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2)
        f.write("\n")
    print(json.dumps(result))
