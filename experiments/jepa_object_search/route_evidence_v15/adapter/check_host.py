"""Freeze a network-denied real-model Habitat smoke test and exact runtime dependencies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def check(project, output):
    folder = Path(__file__).resolve().parent
    v13 = project/"jepa_env_v13/most_integration"
    v14 = project/"jepa_exec_v14"
    train = project/"jepa_evidence_v15/train"
    sys.path[:0] = [str(folder), str(v14), str(v13)]
    from verify_rank import deny_ip_sockets, make_scene, run_case
    deny_ip_sockets()
    from check_policy import check as policy_check
    from v15_agent import EvidenceFrontier
    import jepa_exec_agent
    import numpy as np
    policy_check()
    receipt = json.loads((v13/"no_api_check.json").read_bytes())
    hashes = receipt["sha256"].copy()
    for name, expected in hashes.items():
        assert digest(Path(name)) == expected, name
    weights = [Path(n) for n in receipt["checkpoints"]] + [train/"frozen.pt"]
    dependencies = [
        *weights, train/"evidence_head.py", train/"selection.json", *sorted(folder.glob("*.py")),
        v14/"jepa_exec_agent.py", v14/"verify_rank.py",
    ]
    for p in dependencies:
        hashes[str(p)] = digest(p)
    ranks = []
    native_emit = EvidenceFrontier.emit
    def record(agent, event, **values):
        if event == "rank":
            ranks.append(values.copy())
        return native_emit(agent, event, **values)
    env, episode, dataset = make_scene(project)
    try:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                jepa_exec_agent, "JepaExecutionFrontier", EvidenceFrontier), patch.object(
                    EvidenceFrontier, "emit", record):
            physical = run_case(env, episode, dataset, False, Path(tmp))
    finally:
        env.close()
    rank, = ranks
    assert rank["evidence_policy"] == "goal_evidence_v15" and len(rank["ids"]) == 3
    assert rank["index"] == np.argmax(rank["final_scores"])
    assert np.allclose(rank["final_scores"], np.asarray(rank["v13_scores"])+rank["evidence_bonus"]+rank["exploration_bonus"])
    assert 0 <= rank["v13_index"] < 3 and rank["v13_index"] == rank["v13_same_input_index"]
    keep = ["original","selected","v13_index","index","no_exploration_index","no_evidence_index",
            "v13_scores","evidence_probability","unvisited_fraction","clue_deficit","adaptive_weight",
            "final_scores","evidence_bonus","exploration_bonus"]
    result = dict(passed=True, no_paid_calls=True, causal_frames_only=True, inherited_ip_socket_denial=True,
                  production_model=True, scan_source="real Habitat RGB, simulated native candidate geometry",
                  rank={k:rank[k] for k in keep}, physical=physical,
                  checkpoints=list(map(str, weights)), sha256=hashes)
    # Revalidate the new model and sources following the actual execution.
    for name in [str(p) for p in dependencies]:
        assert digest(Path(name)) == hashes[name]
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(f.name, output)
    finally:
        os.unlink(f.name)
    print(json.dumps(result["rank"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    check(a.project.resolve(), a.output)
