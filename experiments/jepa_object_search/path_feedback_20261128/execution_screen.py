"""Optional conservative execution policy for the frozen mixed path checker."""
from pathlib import Path

import numpy as np

from checker import check_inputs
from mixed_checker import MixedPathChecker


class ExecutionScreen:
    def __init__(self, checker):
        self.checker = checker

    @classmethod
    def load(cls, path):
        return cls(MixedPathChecker.load(path))

    def predict(self, action, path):
        actions, features = check_inputs(action, path)
        size = len(actions)
        length = (actions > 0).sum(axis=1)
        out = {}
        for name, rows in (("near", length <= 36), ("long", length > 36)):
            if not rows.any():
                continue
            limits = self.checker.limits[name]
            upper = limits["admit"] if limits["admit"] is not None else np.nextafter(1., 2.)
            result = self.checker.core.predict(actions[rows], features[rows], threshold=upper)
            probability = result["p_reach"]
            rotation = ~(actions[rows] == 1).any(axis=1)
            accepted = (probability >= upper) | rotation
            result["threshold"][rotation] = min(upper, 1.)
            blocked = np.zeros(len(probability), dtype=bool)
            if limits["block"] is not None:
                blocked = (probability <= limits["block"]) & ~accepted
            status = np.full(len(probability), "uncertain", dtype="<U9")
            status[blocked] = "block"
            status[accepted] = "admit"
            result["screening"] = status
            # An estimated obstruction alone cannot provide a measured side or step.
            advise = blocked & (result["suspect_step"] >= 0)
            result["decision"] = np.where(accepted, "accept", np.where(advise, "replan", "reobserve"))
            result["feedback"] = np.where(accepted, "clear", np.where(advise, "observed_obstacle", "uncertain_path"))
            result["suspect_step"][~advise] = -1
            result["suspect_side"][~advise] = 0
            for key, value in result.items():
                if key not in out:
                    out[key] = np.empty(size, dtype=value.dtype)
                out[key][rows] = value
        return out


def blind_predictions(folder):
    root = Path(__file__).resolve().parent
    model = ExecutionScreen.load(root / "private_checker_mixed/checker.joblib")
    output = root / folder
    output.mkdir(mode=0o700, exist_ok=True)
    files = sorted((root / "private_blind_paired").glob("blind_*.npz"))
    assert len(files) == 6
    for file in files:
        with np.load(file) as data, np.load(root / "private_blind_checker_mixed" / file.name) as reference:
            out = model.predict(data["action"], data["path"])
            assert np.array_equal(out["screening"], reference["screening"])
            assert np.allclose(out["p_reach"], reference["p_reach"])
            assert np.array_equal(out["decision"] == "accept", out["p_reach"] >= out["threshold"])
            assert np.array_equal(out["suspect_step"] >= 0, out["decision"] == "replan")
            np.savez_compressed(output / file.name, group=data["group"], plan_id=data["plan_id"], p_old=reference["p_old"], **out)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        blind_predictions("private_blind_checker_strict")
