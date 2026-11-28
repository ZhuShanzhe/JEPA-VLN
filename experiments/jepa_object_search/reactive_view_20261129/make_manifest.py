"""Record digests of the actual policy, evidence models and paired simulation inputs."""
from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "new_view_20261129"


def fingerprint(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    houses = json.loads((PRIOR / "view_design.json").read_text())["cohorts"]["blind"]
    inputs = {
        "previous_jepa_checkpoint": PRIOR / "private_model/jepa.pt",
        "path_checker": HERE.parent / "path_feedback_20261128/private_checker_mixed/checker.joblib",
        "previous_initial_plans": PRIOR / "private_evaluation/blind_choices.json",
        "sampling_protocol": PRIOR / "view_design.json",
    }
    for name in ("source.npz", "clear.npz", "same.npz", "thresholds.json"):
        inputs["confirm_" + name] = HERE / "private_confirm_models" / name
    for i, house in enumerate(houses, start=1):
        directory = PRIOR / "private_render_blind60" / ("blind_" + house)
        inputs[f"house_{i}_paired_controls"] = directory / "controls.npz"
        inputs[f"house_{i}_source_context"] = PRIOR / "private_data" / ("blind_" + house + ".npz")
    report = {"paired_prior_git_commit": "71c01ea160aaeb75ca2542040517d58dd57a556a",
              "input_sha256": {key:fingerprint(value) for key,value in inputs.items()},
              "implementation_sha256": {p.name:fingerprint(p) for p in sorted(HERE.glob("*.py"))}}
    (HERE / "experiment_manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
