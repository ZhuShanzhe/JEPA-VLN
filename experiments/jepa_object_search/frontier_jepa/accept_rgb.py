"""Compare live context encoding and scores against sealed cached decision-time inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from policy import load_ranker
from vision import ContextEncoder


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error("receipt path must be new")
    torch.set_num_threads(2)
    base = root / "a_exploration" / "paired_branches_v2"
    with (base / "index.jsonl").open() as f:
        # The first indexed dev record, selected before processing any outcome.
        row = next(r for r in map(json.loads,f) if r["split"] == "dev")
    path = base / row["id"]
    with np.load(path / "context.npz") as ctx:
        rgb, xy, plan = (ctx[k] for k in ("rgb", "frontiers_xy", "planned_actions"))
    with np.load(path / "features_vjepa21_b384.npz") as cache:
        cached = cache["context_embedding"]
    encoder = ContextEncoder(root, args.device)
    measured = encoder.encode(rgb)
    np.testing.assert_allclose(measured,cached,rtol=3e-4,atol=3e-4)
    model = load_ranker(args.checkpoint)
    scores = model.score(measured,xy,plan,row["target"])
    reference = model.score(cached,xy,plan,row["target"])
    np.testing.assert_allclose(scores,reference,rtol=1e-4,atol=1e-5)
    assert int(np.argmax(scores)) == int(np.argmax(reference))
    results = dict(raw_context_parity=True, encoder="V-JEPA2.1 ViT-B/16 384 EMA",
                   state=row["id"], goal=row["target"], observed_frames=len(rgb),
                   candidates=len(scores), embedding_size=len(measured), device=args.device,
                   checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                   max_abs_feature_difference=float(np.max(np.abs(measured-cached))),
                   max_abs_score_difference=float(np.max(np.abs(np.array(scores)-reference))),
                   same_candidate_as_cached=True, read_future_images=False, read_teacher_labels=False)
    with args.receipt.open("x") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
