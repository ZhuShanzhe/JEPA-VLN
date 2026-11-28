"""Store or recheck the frozen deployed policy artifacts without storing model weights in Git."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"reactive_view_20261129"


def main(check):
    paths=[HERE/path for path in ("memory_refit.py","clarity_refit.py","refined_inspection.py",
        "private_clarity_model/clear_frame.npz","private_clarity_model/pixel_area.npz","private_clarity_model/calibration.json",
        "private_memory_models/gates.json","private_memory_models/initial.npz",
        "private_memory_models/relook.npz","private_memory_models/foreground.npz")]
    # Include additional learned risk models, when the selected memory policy uses them.
    paths=sorted(set(paths)|set((HERE/"private_memory_models").glob("*.npz")))
    hashes={str(p.relative_to(HERE)):sha256(p.read_bytes()).hexdigest() for p in paths}
    previous=json.loads((HERE/"paired_design.json").read_text())
    assert abs(json.loads((OLD/"private_confirm_models/thresholds.json").read_text())["threshold"]-.835)<1.e-10
    report={"module_hashes":hashes,"unchanged_previous_navigation":previous["input_sha256"],
            "paired_input_sha256":previous["paired_sha256"],"confirmation_threshold":previous["acceptance_product_cutoff"]}
    dest=HERE/"policy_fingerprint.json"
    if check or dest.exists():
        assert json.loads(dest.read_text())==report, "Deployed policy differs from the original paired run."
    else:
        dest.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--check",action="store_true")
    main(p.parse_args().check)
