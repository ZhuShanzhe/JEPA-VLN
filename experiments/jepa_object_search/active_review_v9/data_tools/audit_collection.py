"""Verify independent partitions, co-located camera turns and stable instance labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(root):
    house_union = set()
    result = {}
    cap = np.bincount(np.arange(256) * 7 // 256)
    capacity = cap[:, None] * cap[None, :]
    for split in ["train", "calibration", "holdout"]:
        rows = json.loads((root / (split + ".index.json")).read_bytes())
        houses = set()
        catalog = defaultdict(dict)
        poses = set()
        counts = Counter()
        sources = Counter()
        hashes = []
        for row in rows:
            h = row["house"]
            houses.add(h)
            assert row["split"] == split
            hashes.append(row["sid"])
            with np.load(row["observed"]) as obs, np.load(row["labels"]) as lab:
                assert set(obs.files) == {"rgb", "heading_deg"}
                assert np.array_equal(obs["heading_deg"], np.arange(12)*30)
                rgb = obs["rgb"]
                assert rgb.shape == (12, 256, 256, 3) and rgb.dtype == np.uint8 and rgb.max() > rgb.min()
                pose = (h, *np.round(lab["pose_xyz"], 3).tolist())
                assert pose not in poses
                poses.add(pose)
                src = str(lab["provenance"].item())
                sources[src] += 1
                names = lab["class_name"]
                ids = lab["instance_id"].tolist()
                assert len(ids) == len(set(ids)) == len(names)
                for key, word in zip(ids, names.tolist()):
                    assert catalog[h].setdefault(key, word) == word
                eligible, allpix, near = lab["eligible"], lab["grid"], lab["near_grid"]
                assert eligible.dtype == np.bool_ and eligible.shape == names.shape
                assert allpix.shape == near.shape == (12, len(ids), 7, 7)
                assert (near <= allpix).all() and (allpix.sum(1) <= capacity).all()
                pixels = allpix.sum((2, 3))
                counts["eligible_instance_positions"] += eligible.sum().item()
                counts["eligible_instances_well_visible_some_heading"] += (eligible & (pixels.max(0) >= 64)).sum().item()
                counts["eligible_instances_below64_all_headings"] += (eligible & (pixels.max(0) < 64)).sum().item()
                here = pixels * eligible[None, :]
                switch = (here < 64) & ((np.roll(here, 1, axis=0) >= 64) | (np.roll(here, -1, axis=0) >= 64))
                counts["eligible_instances_cross64_by_one_turn"] += switch.any(axis=0).sum().item()
        assert len(hashes) == len(set(hashes))
        assert not house_union.intersection(houses)
        house_union.update(houses)
        result[split] = dict(houses=len(houses), positions=len(rows), views=12*len(rows), origins=dict(sources),
            **counts, start_ids_sha256=hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest())
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(audit(a.root), indent=2) + "\n")
