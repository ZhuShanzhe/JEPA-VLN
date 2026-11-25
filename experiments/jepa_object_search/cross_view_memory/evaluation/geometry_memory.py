"""Measure what remains for visual memory after an independent depth-location check."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from first_memory import aggregate, calibration_limits, differences, evaluate, make_stream, prepare


def gate(groups, indices, xyz, valid):
    out = {}
    for key, (labels, tables) in groups.items():
        order = indices[key]
        pts = xyz[order]
        dist = np.linalg.norm(pts[:,None] - pts[None,:], axis=-1)
        use = np.isfinite(dist) & valid[order,None] & valid[order][None,:] & (dist <= .60)
        out[key] = labels, {**{tag: np.where(use,sim,-np.inf) for tag,sim in tables.items()},
                            "geometry": np.where(use,-dist,-np.inf)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    args = p.parse_args()
    model = args.root/"model"
    sys.path.insert(0,str(model))
    from reid_model import ViewHead
    manifest = args.root/"data"/"manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    geo = np.load(model/"geometry.npz",allow_pickle=False)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert str(geo["manifest_sha256"].item()) == digest
    ix = geo["ix"].astype(int)
    assert len(np.unique(ix)) == len(ix) and set(ix) == {k for k,r in enumerate(rows) if r["split"] in ("val","test")}
    xyz = np.full((len(rows),3),np.nan)
    valid = np.zeros(len(rows),dtype=bool)
    xyz[ix],valid[ix] = geo["xyz"],geo["valid"]
    assert np.isfinite(xyz[valid]).all()
    torch.set_num_threads(2)
    cached = torch.load(model/"features.pt",weights_only=True,map_location="cpu")
    assert cached["manifest_sha256"] == digest
    heads = {}
    for name in ("direct","jepa"):
        ck = torch.load(model/(name+".pt"),weights_only=True,map_location="cpu")
        heads[name] = ViewHead().eval()
        heads[name].load_state_dict(ck["state_dict"])
    raw, orders = {}, {}
    for split in ("val","test"):
        raw[split] = prepare(rows,cached["features"].float(),split,heads)
        idx = defaultdict(list)
        for k,r in enumerate(rows):
            if r["split"] == split:
                idx[r["house"],r["category"]].append(k)
        orders[split] = {key:make_stream(rows,v) for key,v in idx.items()}
    coverage = {}
    for split in raw:
        chosen = np.array([r["split"]==split for r in rows])
        coverage[split] = dict(available=int((chosen&valid).sum()),total=int(chosen.sum()))
    result = {"protocol":"Matched geometry, correct simulator camera pose, observable RGB-D and object boxes; 0.60 m gate. Missing location abstains. Gaussian stress is illustrative.", "coverage":coverage,"conditions":{}}
    for noise in (0,.25):
        positions = xyz.copy()
        if noise:
            rng = np.random.default_rng(26807)
            positions[:,[0,2]] += rng.normal(0,noise,(len(positions),2))
        groups = {split:gate(raw[split],orders[split],positions,valid) for split in raw}
        methods, test_house = {}, {}
        for tag in ("geometry","frozen","direct","jepa"):
            t = calibration_limits(groups["val"],tag)
            test_house[tag] = evaluate(groups["test"],tag,t)
            methods[tag] = dict(threshold=t,val=aggregate(evaluate(groups["val"],tag,t)),test=aggregate(test_house[tag]))
        result["conditions"][str(noise)] = {"methods":methods,"paired_jepa_minus_comparator":{
            tag:differences(test_house["jepa"],test_house[tag]) for tag in ("geometry","direct")}}
    (args.root/"evaluation"/"geometry_memory_results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(coverage))


if __name__ == "__main__":
    main()
