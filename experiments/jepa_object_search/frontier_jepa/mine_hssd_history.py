"""Mine report-history candidates from existing training houses. Never open dev samples."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "a_exploration" / "paired_branches_v2"


def mine():
    partition = ROOT / "simulation" / "train_dev_v1.json"
    splits = json.loads(partition.read_text())
    assert not set(splits["dev"]) & set(splits["train"])
    counts = Counter()
    records = []
    states_by_house = Counter()
    for manifest_path in sorted((BASE / "train").glob("*/*/*/manifest.json")):
        folder = manifest_path.parent
        m = json.loads(manifest_path.read_text())
        scene = m["source_scene"]
        assert m["split"] == "train" and scene in splits["train"] and scene not in splits["dev"]
        assert m["split_sha256"] == sha(partition)
        judge = json.loads((folder / "teacher" / "labels.json").read_text())
        states_by_house[scene] += 1
        counts["states"] += 1
        all_visible, distinct = set(), []
        for i in range(m["frontier_count"]):
            counts["branches"] += 1
            with np.load(folder / "teacher" / f"branch{i:03d}.npz") as f:
                ids = f["instance_ids"]; pixels = f["target_pixel_counts"]
            assert pixels.shape == (24,len(ids))
            valid = np.flatnonzero(pixels.max(0)>=64)
            all_visible.update(ids[valid].tolist())
            if len(valid) != 1:
                continue
            j = valid[0]
            t = int(np.argmax(pixels[:,j]))
            with np.load(folder / f"branch{i:03d}.npz") as f:
                collisions = f["collided"][:t+1]
                pose = f["relative_pose_xyyaw"][t]
            if np.any(collisions) or np.linalg.norm(pose[:2]) < .45:
                continue
            # These are offline pilot construction criteria, never evaluation filters.
            distinct.append({"branch":i,"instance":int(ids[j]),"peak_pixels":int(pixels[t,j]),"witness_step":t+1})
        counts["states_two_or_more_visible_ids"] += int(len(all_visible)>=2)
        pairs = [[a,b] for a,b in combinations(distinct,2) if a["instance"]!=b["instance"]]
        counts["states_opposite_singletons_collision_free_to_report"] += bool(pairs)
        counts["opposite_branch_pairs"] += len(pairs)
        for pair in pairs:
            records.append({"source":folder.relative_to(BASE).as_posix(),"scene":scene,
                            "target":judge["synset"],"witnesses":pair})
    return {"train_house_count":len(states_by_house),"scan_counts":counts,"train_states_by_house":states_by_house,
            "split_sha256":sha(partition),"source_cohort_sha256":sha(BASE / "cohort.json"),
            "eligibility":"two branches with disjoint singleton target IDs at 64px, each report peak reachable without prior collision, at least .45m from original camera",
            "example_order":"source and branch indices, independent of model scores","pairs":records}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=mine()
    with args.output.open("x") as f:
        json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ("train_house_count","scan_counts")}))
    print(json.dumps(result["pairs"][:1]))
