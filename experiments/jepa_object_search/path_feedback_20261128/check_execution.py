"""Exercise runtime corners with the trained models, without simulator labels or network calls."""
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from execution_screen import ExecutionScreen
from local_repair import consult, select_or_repair
from path_geometry import from_depth, path_features
from sensed_repair import consider


def main():
    root = Path(__file__).resolve().parent
    gate = ExecutionScreen.load(root / "private_checker_mixed/checker.joblib")
    depth = np.full((192, 192), 2.0, dtype=np.float32)
    observed = from_depth(depth)
    # Identical endpoint and yaw, lengths on opposite sides of the operating threshold.
    plans = [(1,) * 25, (2, 3) * 6 + (1,) * 25]
    action = np.zeros((2, 37), dtype=np.int8)
    features = np.zeros((2, 37, 17), dtype=np.float32)
    for i, plan in enumerate(plans):
        action[i, :len(plan)] = plan
        features[i, :len(plan)] = path_features(plan, observed)
    result = gate.predict(action, features)
    expected = [gate.checker.limits[key]["admit"] for key in ("near", "long")]
    assert np.allclose(result["threshold"], expected)
    visuals = dict(sketch=np.zeros((2, 24)), old_joint=np.zeros(2), p_visible_if_reach=np.ones(2))
    assert np.allclose(consider(plans, depth, (), visuals, gate).threshold, expected)
    turns = (2, 3)
    result = gate.predict(np.asarray([turns]), path_features(turns, observed)[None])
    assert result["decision"][0] == "accept" and result["suspect_step"][0] == -1
    empty = np.zeros_like(depth)
    for controller, arg in ((consult, (1, 1)), (select_or_repair, [(1, 1)])):
        reply = controller(arg, empty, np.zeros(2048), None, None, None, None, 36)
        assert reply["status"] == "request_observation" and not reply["actions"]
    print("Passed separate length thresholds, observation loop, rotation and absent-depth checks.")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
