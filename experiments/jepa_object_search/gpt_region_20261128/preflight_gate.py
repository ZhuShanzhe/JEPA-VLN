"""A training-house camera request verifies the private gateway and restart cache."""
import json
from pathlib import Path
import numpy as np
from gpt_gate import GPTGate, MODEL


def main():
    here=Path(__file__).resolve().parent
    path=next(iter(sorted((here.parent/"jepa_env_v13/simulator/trees/train").glob("*/*/node_r.npz"))))
    with np.load(path,allow_pickle=False) as node:
        views=node["ctx"]
    gate=GPTGate("training-transport-probe-v1",here/"private_preflight")
    regions=[dict(id=f"r{i}",distance_m=d,bearing_deg=a,turn_deg=a,steps=8,unseen=.8,
                  blocked=False,prior=.5,novelty=.5) for i,(d,a) in enumerate(((1.5,-30),(2,30)))]
    result=gate.choose(views,"clock",regions,[],{"remaining_steps":300,"step":4})
    assert gate.choose(views,"clock",regions,[],{"remaining_steps":300,"step":4})==result and result["mode"]=="explore"
    receipts=[json.loads(l) for l in gate.path.read_text().splitlines()]
    receipt=next(e for e in reversed(receipts) if e["event"]=="success")
    assert receipts[-1]["event"]=="cache" and receipt["model"]==MODEL
    print(json.dumps(dict(model=MODEL,split="HSSD training",valid_id=True,cache_verified=True,
                          gateway_successes=sum(e["event"]=="success" for e in receipts),
                          input_tokens=receipt["input_tokens"],output_tokens=receipt["output_tokens"])))


if __name__=="__main__":
    main()
