"""Compare frozen JEPA and direct supervision on identical true, ragged HSSD paths."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def summary(rows):
    a = np.asarray([(j, d) for _, j, d in rows], dtype=int)
    groups = defaultdict(list)
    for house,j,d in rows:
        groups[house].append(j-d)
    differences = np.asarray([np.mean(x) for x in groups.values()])
    rng = np.random.default_rng(1616)
    boot = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
    return dict(queries=len(a), houses=len(differences), jepa_hits=int(a[:,0].sum()),
                direct_hits=int(a[:,1].sum()), jepa_only=int((a[:,0]>a[:,1]).sum()),
                direct_only=int((a[:,0]<a[:,1]).sum()),
                equal_house_mean_difference=float(differences.mean()),
                house_bootstrap_95=np.quantile(boot,[.025,.975]).tolist())


def run(project):
    base = project / "jepa_variable_v16/model"
    v15 = project / "jepa_evidence_v15/train"
    sys.path[:0] = [str(base), str(v15), str(project / "jepa_hidden_v11/models")]
    from variable_evidence import VariableEvidence, aggregate, feature_rows
    from calibrate import calibrate, read_logits
    from train_compare import route_inputs
    from evidence_head import EvidenceNet
    torch.set_num_threads(1)
    head = VariableEvidence(project)
    direct = EvidenceNet(head.net.dim)
    direct_path = v15 / "direct.pt"
    direct.load_state_dict(torch.load(direct_path, map_location="cpu", weights_only=True))
    direct.eval()
    logits, y, first, houses, _ = read_logits(project, direct, "calibration")
    assert np.array_equal(first >= 0, y)
    scale, offset = calibrate(logits, first)
    controls = [(head.net, head.scale, head.offset), (direct, scale, offset)]
    eval_dir = project / "jepa_hidden_v11/models/cache/hssd/test/actor"
    paths = {path.stem:path for path in (base / "private_dense").glob("*.npz")}
    reference = json.loads((base / "heldout_dense.json").read_bytes())
    assert len(paths) == reference["origins"]
    results = defaultdict(list)
    for path in sorted(eval_dir.glob("*.npz")):
        with np.load(path) as inputs:
            key = str(inputs["origin"])
            if key not in paths:
                continue
            house = str(inputs["house"])
            assert house not in houses
            actions = inputs["actions"]
            with np.load(paths[key]) as labels:
                assert np.array_equal(actions,labels["actions"]) and np.array_equal(inputs["words"],labels["words"])
                visible = labels["visible"]
            ctx, text = inputs["ctx"].astype(np.float32), torch.from_numpy(inputs["text"].astype(np.float32))
        selected = sorted(range(3),key=lambda i:hashlib.sha256(f"v16-order|{key}|{i}".encode()).digest())
        lengths = [1+hashlib.sha256(f"v16-length|{key}|{i}".encode()).digest()[0]%24 for i in range(3)]
        for condition, ids, steps in [("ragged_2",selected[:2],lengths),("ragged_3",selected,lengths),
                                      ("24",list(range(3)),[24]*3)]:
            candidates = [actions[i,:steps[i]] for i in ids]
            features, taps = feature_rows(ctx, candidates, route_inputs)
            x = torch.from_numpy(features)[None,None]
            for q,goal in enumerate(text):
                true_hits = [bool(visible[i,:steps[i],q].any()) for i in ids]
                hits = []
                for net,s,b in controls:
                    with torch.inference_mode():
                        z = net(x,goal[None])[1].flatten().numpy()
                    _,rate = aggregate(z,taps,s,b)
                    hits.append(int(true_hits[np.argmax(rate)]))
                results[condition].append((house,*hits))
    measured = {k:summary(v) for k,v in results.items()}
    for key, item in measured.items():
        assert item["jepa_hits"] == reference["results"][key]["choice_hits_any_step"]
        assert item["queries"] == reference["queries"] and item["houses"] == reference["houses"]
    output = dict(control_sha256=hashlib.sha256(direct_path.read_bytes()).hexdigest(),
                  calibration="Independent positive-scale hazard calibration for each frozen head, same calibration split",
                  direct_scale=scale,direct_offset=offset,
                  uncertainty="Equal house mean of paired per-query discovery differences, percentile bootstrap "
                              "over 16 houses, seed 1616, 10000 resamples",
                  results=measured)
    return output


if __name__ == "__main__":
    out = Path(sys.argv[2])
    content = json.dumps(run(Path(sys.argv[1]).resolve()), indent=2, allow_nan=False)+"\n"
    if out.exists() and content != out.read_text():
        raise FileExistsError(out)
    out.write_text(content)
    print(content)
