"""Verify splits, inferential units, actual candidate budgets, hidden starts, and fixed-view ambiguity."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def inspect(split, root, external):
    by_hash=defaultdict(list)
    houses=sorted((root/split).glob("tri11_*"))
    membership=[];conditions=defaultdict(Counter)
    matched_views=set();class_feedback=Counter();opportunity=Counter();starts=0;worst_path=0.;worst_return=0.
    for house in houses:
        rec=json.loads((house/"receipt.json").read_bytes())
        assert rec["starts"]==len(list(house.glob("*.npz")))==72
        assert rec["split"]==split and rec["hidden_start_goal_pixels"]==0
        worst_return=max(worst_return,rec["max_return_error_m"])
        membership.append(house.name)
        for f in sorted(house.glob("*.npz")):
            with np.load(f) as z:
                assert z["ctx"].shape==(12,256,256,3) and z["future"].shape==(3,4,256,256,3)
                assert z["initial_target_pixels"].item()==0
                assert np.allclose(z["headings"],np.arange(12)*30)
                assert np.all((z["actions"]==0).sum(1)==20) and z["actions"].shape==(3,28)
                assert z["collided"].shape==(3,28)
                worst_path=max(worst_path,float(abs(z["branch_metres"]-5).max()))
                ids=z["hit_instance_ids"]; hits=z["y"]
                assert np.all((ids>=0).sum(-1)==hits)
                for q, word in enumerate(z["words"]):
                    instance_ids=ids[:,q][ids[:,q]>=0]
                    assert not len(instance_ids) or len(set(instance_ids.tolist()))==len(instance_ids)
                    m=str(z["group"]);n=len(instance_ids)
                    conditions[m][str(n)]+=1
                views=z["ctx"][[11,0,1]]
                digest=hashlib.sha256(views.tobytes()).hexdigest()
                matched_views.add(digest)
                by_hash[(house.name,digest)].append(hits.copy())
                starts+=1
    assert worst_path<.015 and worst_return<.035
    assert not matched_views.intersection(external)
    external.update(matched_views)
    best=0;all_present=0;random=0.;full_oracle=0.
    for key,items in by_hash.items():
        outcomes=np.asarray(items).astype(bool)
        for q in range(2):
            y=outcomes[:,:,q]
            counts=y.sum(0)
            best+=int(counts.max());all_present+=int(y.any(1).sum());random+=float(counts.mean())
            full_oracle+=float(y.max(1).sum())
    return dict(houses=len(houses),configs=starts,hypotheses=starts*2,unique_initial_three_view_inputs=len(matched_views),
        exact_equal_history_first_choice_ceiling=dict(success=best,present=all_present,rate=best/all_present,
            uniform_expected=random/all_present,placement_oracle=full_oracle/all_present),
        condition_counts_by_target_instances={m:dict(z) for m,z in conditions.items()},
        max_distance_error_m=worst_path,max_return_error_m=worst_return)


def verify(root, outfile):
    seen=set()
    result={s:inspect(s,root,seen) for s in ("train","calibration","test")}
    assert [x["houses"] for x in result.values()]==[8,2,6]
    outfile.write_text(json.dumps(result,indent=2,allow_nan=False))
    print("complete",[(k,v["configs"]) for k,v in result.items()])


if __name__=="__main__":verify(Path(sys.argv[1]),Path(sys.argv[2]))
