"""Attribute errors to repeated assets, target categories and context conflicts."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch


def outcome(labels, scores, threshold, mask):
    positive, negative = (labels == 1) & mask, (labels == 0) & mask
    return dict(repeats=int(positive.sum()), recognized=int(((scores >= threshold) & positive).sum()),
                different=int(negative.sum()), wrongly_merged=int(((scores >= threshold) & negative).sum()))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    args = p.parse_args()
    model = args.root / "model"
    sys.path.insert(0, str(model))
    from reid_model import OBJECT_DIM, ViewHead
    torch.set_num_threads(2)
    rows = [json.loads(line) for line in (args.root/"data"/"manifest.jsonl").read_text().splitlines()]
    features = torch.load(model/"features.pt", weights_only=True, map_location="cpu")["features"].float()
    pairs = np.load(model/"pairs_test.npz", allow_pickle=False)
    i, j, y = pairs["i"], pairs["j"], pairs["y"]
    test = [r for r in rows if r["split"] == "test"]
    train_assets = {r["asset"] for r in rows if r["split"] == "train"}
    test_assets = {r["asset"] for r in test}
    templates = np.array([r["asset"] for r in rows])
    categories = np.array([r["category"] for r in rows])
    camera = np.array([r["camera_heading_deg"] for r in rows])
    ids = np.array([r["instance"] for r in rows])
    assert np.array_equal(ids[i] == ids[j], y.astype(bool))
    assert all(rows[a]["split"] == rows[b]["split"] == "test" and rows[a]["house"] == rows[b]["house"]
               and categories[a] == categories[b] for a,b in zip(i,j))
    masks = {"same_asset_physical_impostor": (templates[i] == templates[j]) & (y == 0),
             "different_asset_physical_impostor": (templates[i] != templates[j]) & (y == 0),
             "both_assets_absent_from_train": np.array([a not in train_assets for a in templates[i]]) &
                                               np.array([a not in train_assets for a in templates[j]])}
    for category in ("clock", "wall_art", "picture_frame", "mirror", "chair"):
        masks["category_" + category] = categories[i] == category
    limits = json.loads((model/"metrics.json").read_text())["thresholds"]
    breakdown = {key: {method: outcome(y,pairs["s_"+method],limits[method],mask)
                       for method in ("frozen", "direct", "jepa")} for key, mask in masks.items()}

    # Counterfactual stress probes. The object representation does not change.
    # For a real repeat, borrow a different object's background from this house
    # and category, with camera heading as close as possible. For an
    # impostor, copy the entire stored background as a worst-case confusion test.
    catalog = defaultdict(list)
    for k,row in enumerate(rows):
        if row["split"] == "test":
            catalog[row["house"],row["category"]].append(k)
    donors = {}
    for group in catalog.values():
        for k in group:
            others = [q for q in group if ids[q] != ids[k]]
            if others:
                donors[k] = min(others, key=lambda q: (abs((camera[q]-camera[k]+180) % 360 - 180),q))
    donor_ids = np.array([donors.get(b,b) if same else a for a,b,same in zip(i,j,y)])
    present = np.array([k in donors for k in j]) | (y == 0)
    headings = torch.tensor((camera[j] - camera[i] + 180) % 360 - 180, dtype=torch.float32)
    estimates = []
    head = ViewHead().eval()
    ck = torch.load(model/"jepa.pt", weights_only=True, map_location="cpu")
    head.load_state_dict(ck["state_dict"])
    for k in range(0,len(y),256):
        s = slice(k,k+256)
        candidate = features[j[s]].clone()
        candidate[:,OBJECT_DIM:] = features[donor_ids[s],OBJECT_DIM:]
        logits, _ = head(features[i[s]],candidate,headings[s])
        estimates.append(logits.sigmoid().view(-1).numpy())
    new_scores = np.concatenate(estimates)
    result = {"distinct_test_assets":len(test_assets),"test_assets_present_in_training":len(test_assets & train_assets),
              "paired_subgroups":breakdown,
              "counterfactual_background":{"real_observations":outcome(y,pairs["s_jepa"],limits["jepa"],present),
                                           "background_replaced":outcome(y,new_scores,limits["jepa"],present)}}
    (args.root/"evaluation"/"identity_results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
