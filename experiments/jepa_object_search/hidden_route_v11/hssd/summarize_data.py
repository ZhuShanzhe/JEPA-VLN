"""Report split provenance, physically verified choices and geometry-only choice bias."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def summarize(root,asset):
    cohort=json.loads((root/"cohort.json").read_bytes())
    groups=[{h.split("_")[0] for h in cohort[s]} for s in ("train","calibration","test")]
    history={h.split("_")[0] for houses in cohort["footprints"].values() for h in houses}
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2] or history&groups[2])
    # Check exact rigid-layout repeats in addition to grouping similar numerical IDs.
    house_digest={}
    for p in (asset/"scenes").glob("*.scene_instance.json"):
        objs=json.loads(p.read_bytes()).get("object_instances",[])
        config=sorted(json.dumps(x,sort_keys=True,separators=(",",":")) for x in objs)
        house_digest[p.name.split(".")[0]]=hashlib.sha256("\n".join(config).encode()).hexdigest()
    old=set(cohort["train"]+cohort["calibration"]).union(*(set(x) for x in cohort["footprints"].values()))
    assert not {house_digest[h] for h in cohort["test"]}&{house_digest[h] for h in old if h in house_digest}
    output=dict(cohort_sha256=digest(root/"cohort.json"),same_house_prefix_overlap=0,exact_rigid_layout_hash_overlap=0,
                historic_roots=len(cohort["footprints"]),historical_camera_only_house_count=168,groups={})
    for split in ("train","calibration","test"):
        total=Counter();classes=Counter();per_house={};audit=Counter(); hashes=[];contexts=set()
        positions=np.zeros(3,int);straight=0.;longest=0.;chance=0.
        for house in cohort[split]:
            dst=root/split/house;r=json.loads((dst/"receipt.json").read_bytes())
            assert r["maximum_trials"]==450
            audit.update(json.loads((root/"physical_checks"/(split+"-"+house+".json")).read_bytes()))
            for k in ("attempted","valid_forks_tested","origins","queries"):total[k]+=r[k]
            per_house[house]=dict(origins=r["origins"],queries=r["queries"])
            for p in sorted(dst.glob("*.npz")):
                hashes.append(digest(p))
                with np.load(p) as row:
                    words=row["words"].tolist();y=row["y"]>0
                    a=row["actions"];qcount=len(words);n_distinct=len(set(map(tuple,a)))
                    if n_distinct<3:
                        total["two_unique_action_prefixes_origins"]+=1
                        total["two_unique_action_prefixes_queries"]+=qcount
                    total["multi_instance_house_queries"]+=int((row["house_instances"]>1).sum())
                    total["any_branch_multiple_eligible_instances"]+=int((row["y"].max(0)>1).sum())
                    classes.update(words);positions+=y.sum(1);chance+=y.mean(0).sum()
                    directional=((a==1).astype(float)-(a==2)).cumsum(1)*np.pi/6
                    ahead=((a==0)*np.cos(directional)).sum(1)
                    straight+=y[ahead==ahead.max()].mean(0).sum()
                    forward=(a==0).sum(1)
                    longest+=y[forward==forward.max()].mean(0).sum()
                    contexts.add(hashlib.sha256(row["ctx"].tobytes()).hexdigest())
        count=total["queries"]
        assert len(hashes)==total["origins"]==audit["origins"]==len(contexts) and count==audit["evaluated_queries"]
        output["groups"][split]=dict(counts=total,classes=classes,houses=per_house,physical_replay=audit,
                    positive_count_by_candidate_order=positions.tolist(),candidate_success_random=float(chance/count),
                    prefer_net_forward=float(straight/count),prefer_most_forward_steps=float(longest/count),
                    unique_initial_observations=len(contexts),panels_digest=hashlib.sha256("\n".join(hashes).encode()).hexdigest())
    return output

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("asset",type=Path)
    a=p.parse_args();j=summarize(a.root,a.asset);(a.root/"dataset_summary.json").write_text(json.dumps(j,indent=2))
