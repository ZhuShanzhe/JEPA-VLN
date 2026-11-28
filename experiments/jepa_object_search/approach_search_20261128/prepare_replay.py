"""Reconstruct a frozen HSSD evaluation locally from its digest manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(fh.read()).hexdigest()


def restore(runtime, stage, check_only):
    root = runtime.resolve()
    if any((parent / ".git").exists() for parent in [root, *root.parents]):
        raise ValueError("Use a private runtime outside a Git checkout for the licensed local assets.")
    module = root / "approach_search_20261128"
    panel = module / stage
    prior = root / "prior_approach_20261128"
    owned = root / "route_execution_v1" / "point_executor_hssd_frozen.py"
    slots = {name: module / name for name in ("map_views.py", "global_frontiers.py", "search_policy.py", "enhanced_search.py")}
    slots.update({name: prior / name for name in ("placement_prior.py", "context_calibration.json", "learned_prior.json")})
    slots["point_executor.py"] = owned
    for name in ("__init__.py", "navigator.py"):
        slots["agents/" + name] = root / "private_sources" / "MOST-Nav" / "agents" / name
    lock = json.loads((panel / "locked_sources.json").read_text())
    pending = []
    for rel, digest in lock.items():
        dst = (root / rel).resolve()
        if not dst.is_relative_to(root):
            raise ValueError(f"Invalid manifest path: {rel}")
        if dst.exists():
            src = dst
        elif dst.is_relative_to(panel / "locked"):
            slot = dst.relative_to(panel / "locked").as_posix()
            src = slots[slot]
            pending.append((src, dst))
        else:
            raise FileNotFoundError(rel)
        if not src.is_file() or sha256(src) != digest:
            raise ValueError(f"Missing or different frozen dependency for {rel}: {src}")
    if not check_only:
        for src, dst in pending:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    print(f"{stage}: verified {len(lock)} files, {len(pending)} snapshot files {'to materialize' if check_only else 'materialized'}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("runtime", type=Path)
    cli.add_argument("stage", choices=["eval3", "eval4"])
    cli.add_argument("--check-only", action="store_true")
    args = cli.parse_args()
    restore(args.runtime, args.stage, args.check_only)
