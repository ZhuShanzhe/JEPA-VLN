"""Evaluate local pose correction on held-out camera groups with target labels read only after selection."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np

from refine_view import correct_view


def evaluate(records):
    totals=Counter()
    by_house=defaultdict(Counter)
    for pose,r,j,reach,visible,capable,safe,house,in_fov,intercepted in records:
        idx=np.flatnonzero(capable & safe)
        if not len(idx):
            totals["groups_without_safe_camera"]+=1
            continue
        totals["groups_evaluated"]+=1
        rows=Counter()
        # Treat each available camera as a possible initial proposal. Counts share source views.
        for i in idx:
            dst=correct_view(pose,r,j,capable & safe,i)
            assert dst is not None and capable[dst] and safe[dst]
            if reach[i] < 0 or reach[dst] < 0:
                rows["missing_labels"]+=1
                continue
            if (reach[i] and visible[i]<0) or (reach[dst] and visible[dst]<0):
                rows["missing_labels"]+=1
                continue
            before=bool(reach[i] and visible[i])
            after=bool(reach[dst] and visible[dst])
            rows["initial_proposals"]+=1
            rows["joint_before"]+=int(before)
            rows["joint_after"]+=int(after)
            rows["proposed_change"]+=int(i!=dst)
            rows["rescued"]+=int(after and not before)
            rows["spoiled"]+=int(before and not after)
            if i!=dst:
                rows["changed_translation"]+=int(np.linalg.norm(pose[dst,:2]-pose[i,:2])>1e-4)
                rows["changed_height"]+=int(abs(pose[dst,2]-pose[i,2])>1e-4)
                rows["changed_yaw_or_pitch"]+=int(np.linalg.norm(pose[dst,4:8]-pose[i,4:8])>1e-4)
            if reach[i] and visible[i]==0:
                rows["reachable_but_invisible"]+=1
                rows["invisible_rescued"]+=int(after)
                if in_fov[i]>=4 and intercepted[i]>=.5*in_fov[i]:
                    rows["confirmed_interception"]+=1
                    rows["confirmed_interception_rescued"]+=int(after)
        totals.update(rows)
        by_house[house].update(rows)
    result=dict(counts=dict(totals),house_deltas=[
        {"initial_proposals":v["initial_proposals"],"joint_after_minus_before":v["joint_after"]-v["joint_before"]}
        for k,v in sorted(by_house.items())])
    if by_house:
        changes=[(v["joint_after"]-v["joint_before"])/v["initial_proposals"] for v in by_house.values()
                 if v["initial_proposals"]]
        if changes:
            draws=np.random.default_rng(1128).choice(changes,(10000,len(changes)),replace=True).mean(axis=1)
            result["mean_house_change"]=float(np.mean(changes))
            result["house_bootstrap_interval95"]=np.quantile(draws,[.025,.975]).tolist()
    return result


def run(predictions, data_root):
    p=np.load(predictions,allow_pickle=False)
    native,experimental=[],[]
    for name in np.unique(p["source_file"]):
        # Artifacts can be regenerated in a different project location.
        path=Path(str(name))
        if not path.is_file():
            paths=list(data_root.rglob(path.name))
            assert len(paths)==1,(path.name,len(paths))
            path=paths[0]
        with np.load(path,allow_pickle=False) as src:
            for group in np.unique(p["group"][p["source_file"]==name]):
                ix=np.flatnonzero((p["source_file"]==name)&(p["group"]==group))
                rows=np.asarray(p["source_row"][ix],dtype=int)
                pose=p["pose"][ix]
                assert np.allclose(pose,src["pose"][rows])
                r=p["p_reach"][ix].reshape(-1)
                j=p["p_joint"][ix].reshape(-1)
                reach=p["reach"][ix].reshape(-1)
                visible=p["visible"][ix].reshape(-1)
                safe=src["safe_mask"][rows].astype(bool).reshape(-1)
                deployed=src["native_mask"][rows].astype(bool).reshape(-1)
                occlusion=(src["in_fov"][rows].reshape(-1),src["intercepted"][rows].reshape(-1))
                assert len(set(map(str,p["house"][ix])))==1
                house=str(p["house"][ix][0])
                args=(pose,r,j,reach,visible)
                native.append((*args,deployed,safe,house,*occlusion))
                experimental.append((*args,np.ones_like(safe),safe,house,*occlusion))
    return dict(protocol="Local pose proposals in held-out cue groups. Current source-depth clearance filter; no ground truth in selection. Camera changes restricted to a 1.5m horizontal, 0.5m vertical, 90deg yaw and 40deg pitch neighborhood. Joint threshold 0.5, minimum predicted improvement 0.05. All available supported source proposals tested, dependent within group.",
                fixed_camera=evaluate(native),offline_camera_variation=evaluate(experimental))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("predictions",type=Path)
    parser.add_argument("data_root",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    result=run(args.predictions,args.data_root)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v["counts"] for k,v in result.items() if isinstance(v,dict)}))
