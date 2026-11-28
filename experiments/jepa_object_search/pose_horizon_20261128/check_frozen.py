"""Verify the production discovery gate and route scorer against the preceding accepted release."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RELEASE = "d3c51bfd902fa213458628381245617967ff1762"
REPO = ROOT / "owned_publication_20261128"
PREFIX = "experiments/jepa_object_search"
FILES = {
    "observation_route_20261128": ["gateway.py", "policy.py", "routes.py", "terminal_prefix.py",
                                  "host.py", "host_dispatch.py"],
    "multiview_nav_20261128": ["multi_worker.py", "multi_gate.py", "multi_policy.py"],
}


def verify():
    checked = {}
    for directory, names in FILES.items():
        for name in names:
            local = ROOT / directory / name
            expected = subprocess.check_output(["git","-C",str(REPO),"show",f"{RELEASE}:{PREFIX}/{directory}/{name}"])
            current = local.read_bytes()
            assert current.replace(b"\r\n",b"\n") == expected.replace(b"\r\n",b"\n"), str(local)
            checked[f"{directory}/{name}"] = hashlib.sha256(current).hexdigest()
    heads = {}
    inputs=[ROOT/"jepa_evidence_v15/train/frozen.pt", ROOT/"jepa_variable_v16/model/calibration.json",
            ROOT/"jepa_env_v13/models/selection.json", ROOT/"jepa_env_v13/models/prior.json",
            ROOT/"local_target_20261128/localvision/surface_reader.npz"]
    inputs.extend(sorted((ROOT/"jepa_env_v13/models/weights").glob("jepa_11_*.pt")))
    assert len(inputs) >= 6
    for file in inputs:
        heads[file.relative_to(ROOT).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    return dict(compared_release=RELEASE, deployed_sources=checked, deployed_parameters=heads)


if __name__=="__main__":
    receipt=HERE/"frozen_contract.json"
    now=verify()
    if receipt.exists():
        assert json.loads(receipt.read_text())==now
    else:
        receipt.write_text(json.dumps(now,indent=2)+"\n")
    print("Verified pre-existing production gate, candidate construction, ranking and dispatch.")
