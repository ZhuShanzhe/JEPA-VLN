"""Recreate the frozen training-house probe, in a separate per-arm provider journal."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "gpt_region_20261128"
sys.path.insert(0, str(OLD))
from gpt_gate import GPTGate, MODEL


def receipts(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    lock = json.loads((OLD/"gate_lock.json").read_text())
    code_hash = sha256((OLD/"gpt_gate.py").read_bytes()).hexdigest()
    assert code_hash == lock["gpt_region_20261128/gpt_gate.py"]
    images = sorted((HERE.parent/"jepa_env_v13/simulator/trees/train").glob("*/*/node_r.npz"))[0]
    with np.load(images, allow_pickle=False) as node:
        views = node["ctx"]
    regions = [dict(id=f"r{i}",distance_m=d,bearing_deg=a,turn_deg=a,steps=8,unseen=.8,
                    blocked=False,prior=.5,novelty=.5) for i,(d,a) in enumerate(((1.5,-30),(2,30)))]
    new = GPTGate("training-transport-probe-v1", HERE/"private_preflight")
    answer = new.choose(views, "clock", regions, [], {"remaining_steps":300,"step":4})
    assert new.choose(views, "clock", regions, [], {"remaining_steps":300,"step":4}) == answer
    current = receipts(new.path)
    past = receipts(OLD/"private_preflight"/new.path.name)
    current_success, = (r for r in current if r["event"]=="success")
    past_success, = (r for r in past if r["event"]=="success")
    assert current_success["hash"] == past_success["hash"]
    assert current_success["model"] == past_success["model"] == MODEL
    assert current[-1]["event"]=="cache"
    result=dict(split="HSSD training",model=MODEL,provider_client_sha256=code_hash,
                identical_request_hash_to_previous_arm=True,isolated_receipt_directory=True,
                one_paid_success=True,replay_cache_verified=True,
                same_mode_as_previous_probe=answer["mode"]==past_success["decision"]["mode"],
                input_tokens=current_success["input_tokens"],output_tokens=current_success["output_tokens"])
    (HERE/"gate_input_receipt.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    main()
