"""Evaluate ragged evidence predictions on frozen real HSSD route prefixes."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from variable_evidence import VariableEvidence


def main(project):
    sys.path.insert(0, str(project / "jepa_hidden_v11/models"))
    from train_compare import route_inputs
    torch.set_num_threads(1)
    base = project / "jepa_variable_v16/model"
    model = VariableEvidence(project)
    root = project / "jepa_hidden_v11/models/cache/hssd/test/actor"
    dense = {p.stem: p for p in (base / "private_dense").glob("*.npz")}
    expected_origins, houses = set(), set()
    for p in (project / "jepa_hidden_v11/hssd/test").glob("*/private_replay.json"):
        ordered = sorted([x["origin"] for x in json.loads(p.read_text())],
                         key=lambda s: hashlib.sha256(("v16-dense|" + s).encode()).digest())
        expected_origins.update(ordered[:5])
        if ordered:
            houses.add(p.parent.name)
    if set(dense) != expected_origins:
        raise RuntimeError(f"waiting for physical trajectories ({len(dense)}/{len(expected_origins)})")
    totals = defaultdict(lambda: defaultdict(float))
    sizes, query_count = [], 0
    for p in sorted(root.glob("*.npz")):
        with np.load(p) as actor:
            origin = str(actor["origin"])
            if origin not in dense:
                continue
            with np.load(dense[origin]) as d:
                vis = d["visible"]
                assert np.array_equal(actor["actions"], d["actions"])
                assert actor["words"].tolist() == d["words"].tolist()
            ctx, plans = actor["ctx"].astype(np.float32), actor["actions"]
            text = torch.from_numpy(actor["text"].astype(np.float32))
        mixed = [1+hashlib.sha256(f"v16-length|{origin}|{k}".encode()).digest()[0]%24 for k in range(len(plans))]
        order = sorted(range(len(plans)), key=lambda k: hashlib.sha256(f"v16-order|{origin}|{k}".encode()).digest())
        query_count += len(text)
        configs = [(str(t), list(range(3)), [t]*3) for t in (1,2,3,4,5,6,8,12,18,24)]
        for k in (2,3):
            ids = order[:k]
            configs.append((f"ragged_{k}", ids, [mixed[j] for j in ids]))
        for name, ids, lengths in configs:
            lengths = np.array(lengths)
            choices = [plans[i,:t] for i,t in zip(ids, lengths)]
            d = totals[name]
            for q, goal in enumerate(text):
                out = model.predict_features(ctx, goal[None], choices, route_inputs)
                truth = np.asarray([bool(vis[k,np.array(taps)-1,q].any())
                                    for k,taps in zip(ids,out["tap_steps"])])
                continuous = np.asarray([bool(vis[k,:n,q].any()) for k,n in zip(ids,lengths)])
                pred = np.asarray(out["evidence_probability"])
                best = int(np.argmax(out["evidence_rate"]))
                d["queries"] += 1
                d["candidate_predictions"] += len(choices)
                d["candidate_hits_at_readouts"] += int(truth.sum())
                d["candidate_hits_any_step"] += int(continuous.sum())
                d["choice_hits_at_readouts"] += int(truth[best])
                d["choice_hits_any_step"] += int(continuous[best])
                d["random_hits_any_step"] += continuous.mean()
                d["oracle_hits_any_step"] += continuous.any()
                d["brier_sum"] += ((truth-pred)**2).sum()
                d["evaluated_steps"] += lengths.sum()
                if name == "ragged_3":
                    sizes.extend(lengths.tolist())
    assert query_count > 0
    for d in totals.values():
        d["brier_at_readouts"] = d.pop("brier_sum") / d["candidate_predictions"]
        for k in list(d):
            if k not in ("brier_at_readouts", "random_hits_any_step"):
                d[k] = int(d[k])
    result = dict(houses=len(houses), origins=len(dense), queries=query_count,
                  actual_action_steps_replayed=len(dense)*3*24,
                  ragged_lengths_observed=sorted(set(sizes)), results=totals,
                  selection="up to five origins per V11 held-out house by stable SHA-256 before reading outcomes",
                  actor="V16 duration-normalized evidence only; no online memory or planner",
                  readouts="actual observations at multiples of six plus a final partial endpoint",
                  any_step="a real target instance occupies at least 32 pixels within 2m at any retained step")
    path = base / "heldout_dense.json"
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != content:
        raise FileExistsError(path)
    path.write_text(content)
    print(json.dumps({k:v for k,v in result.items() if k != "results"}))
    for name in ("3","6","24","ragged_2","ragged_3"):
        print(name, dict(totals[name]))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
