"""Reconstruct per-route counts from instance hit tensors and report category/scene coverage."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path

import numpy as np


@lru_cache(maxsize=3)
def geometry(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in ("ids","seen","start","hits","categories","plans")}


def audit(path):
    rows = json.loads(path.read_bytes())
    stats, classes, starts = Counter(), defaultdict(Counter), defaultdict(set)
    seen_keys = set()
    for r in rows:
        geo = geometry(r["geometry"])
        h = r["horizon"]
        key = (r["sid"],h,r["goal_category"])
        assert key not in seen_keys
        seen_keys.add(key)
        assert r["reported_ids_before"] == [] and np.unique(geo["ids"]).size == len(geo["ids"])
        assert np.array_equal(geo["plans"][:,:h],r["action"])
        mask = geo["categories"] == r["goal_category"]
        assert mask.any()
        hit = geo["hits"][:,:h,mask].any(axis=1)
        new = hit & ~geo["start"][mask]
        unseen = new & ~geo["seen"][mask]
        gain = new.sum(axis=1)
        assert np.array_equal(gain,r["gain_newly_eligible"])
        assert np.array_equal(unseen.sum(axis=1),r["gain"])
        assert np.array_equal(hit.sum(axis=1),r["gain_unreported_only"])
        assert set(r["start_eligible_ids"]) == set(geo["ids"][mask][geo["start"][mask]].tolist())
        flags = dict(rows=1, any_new=int(gain.max()>0), informative=int(gain.max()>gain.min()),
                     zero=int(gain.max()==0), includes_previously_seen=int(np.any(new & geo["seen"][mask])))
        stats.update(flags)
        classes[r["goal_category"]].update(flags)
        for name, active in flags.items():
            if active: starts[name].add(r["sid"])
    return {"houses":len({r["house"] for r in rows}),"starts":len(starts["rows"]),
            "informative_starts":len(starts["informative"]),"starts_with_seen_but_new":len(starts["includes_previously_seen"]),
            "rows":dict(stats),"categories":{k:dict(v) for k,v in sorted(classes.items())}}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--official",type=Path,required=True)
    ap.add_argument("--extended",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    report={}
    catalogs={}
    for label, base in (("official",a.official),("derived",a.extended)):
        report[label]={}
        catalogs[label]={}
        for split in ("train","calibration","holdout"):
            f=base/(split+".rows.json")
            report[label][split]=audit(f)
            catalogs[label][split]={r["house"] for r in json.loads(f.read_bytes())}
    assert catalogs["official"]==catalogs["derived"]
    pools=list(catalogs["official"].values())
    assert sum(map(len,pools))==len(set.union(*pools))
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
