"""Fingerprint the paired inputs and frozen navigation code before the new confirmation evaluation."""
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "reactive_view_20261129"
PRIOR = HERE.parent / "new_view_20261129"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cohort = json.loads((PRIOR / "view_design.json").read_text())["cohorts"]
    splits = [set(cohort[k]) for k in ("train", "val", "blind")]
    assert list(map(len,splits)) == [12,6,3] and sum(map(len,splits)) == len(set.union(*splits))
    counts, caps = Counter(), Counter()
    trajectory = hashlib.sha256()
    for house in cohort["blind"]:
        for i in range(60):
            path = OLD / "private_roll" / house / f"{i}.json"
            r = json.loads(path.read_text())
            if not r["focus"]:
                continue
            trajectory.update((house + "/" + str(i) + ":" + sha(path) + "\n").encode())
            t, e = r["new_confirmation"], r["events"]
            if t is not None:
                counts["correct" if e[t]["correct"] else "false_report"] += 1
            else:
                counts["clear_unreported" if any(f["px"] >= 369 for f in e) else "never_clear_unreported"] += 1
            caps[int(r["budget"])] += 1
    assert counts == {"correct":26,"false_report":6,"clear_unreported":29,"never_clear_unreported":33}
    inputs = {
        "original_route_head":PRIOR / "private_model/jepa.pt",
        "passability_checker": HERE.parent / "path_feedback_20261128/private_checker_mixed/checker.joblib",
        "initial_route":PRIOR / "observation_actor.py", "recovery_route":OLD / "reactive_route.py",
        "motion_loop":OLD / "inspection_session.py", "original_threshold":OLD / "private_confirm_models/thresholds.json",
        "cohort_design":PRIOR / "view_design.json"
    }
    threshold = json.loads(inputs["original_threshold"].read_text())["threshold"]
    assert abs(threshold - .835) < 1e-8
    report = {
        "baseline_commit":"9780d69ec494f60961325828008af697c7d0d84a",
        "set":"Same 94 previously evaluated local re-observation tasks across three held-out HSSD houses.",
        "budget_counts":dict(caps),"previous_outcomes":dict(counts),
        "split_houses":{"train":12,"calibration":6,"evaluation":3},
        "input_sha256":{k:sha(v) for k,v in inputs.items()},
        "paired_sha256":trajectory.hexdigest(),"acceptance_product_cutoff":threshold,
        "primary_criterion":"First submitted frame only. >=369 target pixels; box covers >=50% of them and IoU against target envelope >=0.30.",
        "endpoints":["correct new first submissions among 29 prior abstentions despite visible opportunity",
                     "losses among 26 correct prior first submissions",
                     "paired new, removed and persistent false first submissions among all 94"],
        "actions":"Identical action prefix until first confirmation; real acquisition, turns and collisions consume the existing per-request budget. Pending memory consumes the next distinct view on the existing controller.",
        "selection":"Train and calibrate only in separate houses. Freeze candidate before evaluating this development panel. No threshold sweep over panel outcomes."
    }
    path = HERE / "paired_design.json"
    payload = json.dumps(report,indent=2,sort_keys=True) + "\n"
    if path.exists() and path.read_text()!=payload:
        raise RuntimeError("Previous audit or frozen navigation inputs have changed")
    path.write_text(payload)
    print(dict(counts),dict(caps),trajectory.hexdigest())


if __name__=="__main__":
    main()
