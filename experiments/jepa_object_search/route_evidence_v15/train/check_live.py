"""Check the frozen CPU RGB entry against actor caches from eight distinct test houses."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from evidence_head import EvidencePredictor


def main(p):
    from types import SimpleNamespace
    torch.set_num_threads(2)
    sys.path.insert(0,str(p/"jepa_hidden_v11/models"))
    from encode import Encoder
    from train_compare import route_inputs
    model=SimpleNamespace(encoder=Encoder(p),old=SimpleNamespace(route_inputs=route_inputs))
    assert model.encoder.device == "cpu"
    pred=EvidencePredictor(p)
    v11=p/"jepa_hidden_v11"
    test=v11/"hssd/test"
    cache=v11/"models/cache/hssd/test/actor"
    results=[]
    for house in sorted(d for d in test.iterdir() if d.is_dir())[:8]:
        file=next(iter(sorted(house.glob("*.npz"))))
        key=hashlib.sha256(str(file.relative_to(v11)).encode()).hexdigest()
        with np.load(file) as z:
            rgb=z["ctx"]
            plans=z["actions"]
            goal=str(z["words"][0])
        with np.load(cache/(key+".npz")) as z:
            assert goal == z["words"][0] and np.array_equal(plans,z["actions"])
            x=route_inputs(z["ctx"].astype("float32"),plans)
            text=torch.from_numpy(z["text"][:1])
        with torch.inference_mode():
            cached=pred.net(torch.from_numpy(x)[None],text)[0][0].numpy()
            live=pred.predict(model,rgb,goal,plans)
        results.append(dict(house=house.name,maximum_absolute_probability_difference=float(np.max(np.abs(cached-live))),
                            same_selected_route=bool(live.argmax()==cached.argmax()), cached=cached.tolist(),live=live.tolist()))
    groups = {}
    for split, exp in (("train","jepa_hidden_v11"),("extra_train","jepa_route_v12"),
                       ("calibration","jepa_hidden_v11"),("test","jepa_hidden_v11"),("fresh2","jepa_route_v12")):
        groups[split]=set()
        for file in (p/exp/"models/cache/hssd"/split/"actor").glob("*.npz"):
            with np.load(file) as z:
                groups[split].add(str(z["house"]))
    pairs={a+"_"+b:len(groups[a]&groups[b]) for i,a in enumerate(groups) for b in list(groups)[i+1:]}
    receipt=dict(cpu_only=True,sampling="lexicographic first file and first goal in each of first 8 test houses",
                 cases=results,max_error=max(r["maximum_absolute_probability_difference"] for r in results),
                 matched=sum(r["same_selected_route"] for r in results), split_overlaps=pairs)
    out=p/"jepa_evidence_v15/train/live_receipt.json"
    out.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))


if __name__=="__main__":
    main(Path(sys.argv[1]))
