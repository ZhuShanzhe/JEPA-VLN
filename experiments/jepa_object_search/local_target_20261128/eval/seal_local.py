"""Verify the old system and lock the new arm before its first evaluation rollout."""
from pathlib import Path
import argparse
import hashlib
import json

HERE=Path(__file__).resolve().parent
MODULE=HERE.parent
PROJECT=MODULE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",type=Path)
    args=parser.parse_args()
    stage=args.stage.resolve()
    base=PROJECT/"approach_search_20261128/eval4"
    paths=[]
    for manifest in (stage/"panel_lock.json",base/"locked_sources.json"):
        frozen=json.loads(manifest.read_text())
        for rel,digest in frozen.items():
            assert sha(PROJECT/rel)==digest,rel
        paths.extend(PROJECT/rel for rel in frozen)
        paths.append(manifest)
    paths.extend(MODULE/name for name in ("local_search.py","local_worker.py","local_runtime.py","waypoints/observation_sites.py",
                                          "localvision/local_vision.py","eval/run_local.py","eval/seal_local.py"))
    visual=MODULE/"localvision"
    if (visual/"surface_reader.npz").exists():
        report=json.loads((visual/"surface_validation.json").read_text())
        assert sha(visual/"surface_reader.npz")==report["selected_sha256"]
        families=lambda values: {v.split("_")[0] for v in values}
        evaluation=families(r["scene"] for r in json.loads((stage/"panel.json").read_text()))
        assert not evaluation & families(report["train_houses"]+report["check_houses"])
        paths.extend([visual/"surface_reader.npz",visual/"surface_validation.json"])
    paths.extend((PROJECT/"jepa_env_v13/models").glob("weights/*.pt"))
    paths.extend([PROJECT/"jepa_env_v13/models/env_routes.py",PROJECT/"jepa_env_v13/models/prior.json"])
    values={p.relative_to(PROJECT).as_posix():sha(p) for p in sorted(set(paths))}
    out=stage/"local_lock.json"
    if out.exists():
        assert json.loads(out.read_text())==values
    else:
        assert not list((stage/"runs").glob("*_local.json"))
        with out.open("x") as target:
            json.dump(values,target,indent=2)
    print(len(values),sha(out))


if __name__=="__main__":
    main()
