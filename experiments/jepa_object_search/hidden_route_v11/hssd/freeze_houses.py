"""Construct a conservative new-house split from prior project run footprints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import yaml


def freeze(disk, dest):
    base = disk/"jepa_object_search"
    assets = disk/"jepa_vln_data_20261014/raw/hssd-hab"
    splits = yaml.safe_load((assets/"scene_splits.yaml").read_text())
    train = set(splits["train"])
    rx = re.compile(rb"(?<![a-zA-Z0-9])(?:10[0-9]{7}(?:_[0-9]{9})?)(?![a-zA-Z0-9])")
    roots = [p for p in base.iterdir() if (p.name.startswith(("jepa_five_","jepa_separation_","jepa_verification_","jepa_memory_","v9_","training_round1_")) or p.name in ("frontier_jepa","training_ready","comparisons_nov14")) and p.is_dir()]
    roots += [base/"a_exploration"/n for n in ("paired_branches_v1","paired_branches_v2","hssd_smoke","official_train_smoke")]
    roots += [disk/d/"results" for d in ("jepa_vln_baselines_20261015","jepa_instance_bench_20261016","jepa_vln_paper_bench_20261017")]
    roots += [base/"mapping_runtime/pilots",base/"b_verification",base/"a_hssdov_external/runtime_check"]
    footprints = {}
    for root in roots:
        found = set()
        for parent, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if x not in (".git","node_modules","__pycache__", "site-packages","repos","downloads","policy_env")]
            for field in (parent, *files):
                found.update(x.decode() for x in rx.findall(str(field).encode()))
            for f in files:
                path = Path(parent)/f
                if path.suffix.lower() not in (".json",".jsonl",".csv",".yaml",".log",".tsv"):continue
                if path.stat().st_size>20_000_000:continue
                if f in ("query_map.json","camera_extrinsics_audit.json") or any(t in f.lower() for t in ("catalog","eligible","plan.json","split","transfer_inventory")):continue
                found.update(x.decode() for x in rx.findall(path.read_bytes()))
        footprints[str(root.relative_to(disk))]=sorted(found&train)
    blocked=set.union(set(),*(set(s) for s in footprints.values()))
    protected=set(splits["val"])
    panels=json.loads((base/"simulation/scene_partitions.json").read_bytes())
    for x in panels["langnav_public_scenes"].values():protected.update(x)
    protected.update(p.name.split(".json")[0] for p in (base/"a_hssdov_external/eval/hssdov/content").glob("*.json.gz"))
    token="hssd-hidden-20261124-physical-v11"
    order=lambda ids:sorted(ids,key=lambda x:hashlib.sha256((token+x).encode()).digest())
    available={p.name.split(".json")[0] for p in (base/"a_exploration/official/hssd_v025/train/content").glob("*.json.gz")}
    prefix=lambda v:v.split("_")[0]
    forbidden={prefix(x) for x in blocked|protected}
    def unique_prefix(values):
        rows={}
        for x in order(values):rows.setdefault(prefix(x),x)
        return list(rows.values())
    fresh=unique_prefix(x for x in (train&available) if prefix(x) not in forbidden)
    seen=unique_prefix((train&available)&blocked-protected)
    out=dict(official_train=len(train),official_val=len(splits["val"]),footprints=footprints,protected=sorted(protected),
             fresh_count=len(fresh),fresh_candidates=fresh,train=seen[:28],calibration=fresh[16:22],
             test=fresh[:16],salt=token)
    if len(fresh)<22:
        test_count=min(16,max(1,len(fresh)-4));out["test"]=fresh[:test_count]
        out["calibration"]=fresh[test_count:test_count+6] or seen[28:34]
    dest.write_text(json.dumps(out,indent=2))
    print("fresh",len(fresh),"train",len(out["train"]),"cal",len(out["calibration"]),"test",len(out["test"]),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("disk",type=Path);p.add_argument("dest",type=Path)
    a=p.parse_args();freeze(a.disk,a.dest)
