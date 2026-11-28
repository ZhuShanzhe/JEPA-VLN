"""Paired local-waypoint experiment. Frozen decision code receives no outcome or instance masks."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from view_dispatch import fallback, shortest_per_view
from view_model import current_inputs

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / "path_feedback_20261128"
PRE = ROOT.parent / "pose_horizon_20261128"
sys.path.extend([str(LEGACY), str(PRE)])
from local_repair import select_or_repair
from mixed_checker import MixedPathChecker
from observation_head import HorizonCameraPredictor
from path_geometry import from_depth, path_features
from replan import RepairScorer


def assess(gate, plans, depth):
    length = max(map(len, plans))
    actions = np.zeros((len(plans), length), np.uint8)
    features = np.zeros((len(plans), length, 17), np.float32)
    geometry = from_depth(depth)
    for i, p in enumerate(plans):
        actions[i, :len(p)] = p
        features[i, :len(p)] = path_features(p, geometry)
    out = gate.predict(actions, features)
    return out["decision"] == "accept", out["p_reach"]


def infer(bl, d, pred, source):
    gate, horizon, scorer = bl
    plans = [tuple(map(int,a[a != 0])) for a in d["action"]]
    lens = np.array([len(p) for p in plans])
    kind = d["kind"].astype(str)
    view = d["view_id"]
    old = np.flatnonzero(kind == "old")
    old = old[np.argsort(d["plan_id"][old])]
    assert np.array_equal(d["plan_id"][old], np.arange(7))
    left, depth = source["left_depth"], source["depth"]
    budget = int(d["budget"][0] - d["scan_cost"][0])
    assert np.all(lens[old] <= budget)
    offered = [plans[i] for i in old]
    pose_by_plan = {plans[i]: d["pose"][i] for i in old}
    op = select_or_repair(offered[:3], depth, d["context"][old[0]], pose_by_plan.__getitem__,
                          gate,horizon,scorer,budget)
    baseline = int(old[offered.index(tuple(op["actions"]))]) if op["status"] == "execute" else None
    ok, pr = assess(gate, plans, depth)
    assert baseline is None or ok[baseline]
    allowed_old = old[ok[old]]
    choices = {"old": baseline,"old_all_seven":int(allowed_old[np.argmin(lens[allowed_old])]) if len(allowed_old) else None}
    rel = [(3,) + p for p in offered]  # turn right back from the shared 30-degree left scan
    if baseline is None:
        left_ok, _ = assess(gate, rel, left)
        left_allowed = old[left_ok]
        scan_choice = int(left_allowed[np.argmin(lens[left_allowed])]) if len(left_allowed) else None
    else:
        scan_choice = baseline
    choices["old_second_look"] = scan_choice
    switching = baseline is None and not len(allowed_old)
    for name, key in (("jepa","p_visible_jepa"),("direct","p_visible_direct"),
                      ("jepa_no_map","p_visible_no_map"),("prior_visible","p_visible_frozen")):
        if key in pred:
            choices[name] = fallback(baseline, kind, view, ok, lens, budget, pr, pred[key])
    _, geom_logit = current_inputs(d["context"],d["pose"],d["map"])
    choices["projected_area"] = fallback(baseline,kind,view,ok,lens,budget,pr,1./(1.+np.exp(-geom_logit)))
    reps = shortest_per_view(view, ok & (kind=="alt") & (lens<=budget), lens)
    choices["nearest"] = int(reps[np.argmin(lens[reps])]) if switching and len(reps) else baseline
    return choices, ok, switching, old, lens


def interval(groups, metrics):
    houses = sorted(groups)
    rng = np.random.default_rng(2911)
    draws = rng.integers(len(houses), size=(10000, len(houses)))
    out = {}
    for comparison in metrics:
        deltas = np.array([(groups[h]["jepa"]["clear"] - groups[h][comparison]["clear"]) /
                           groups[h]["jepa"]["groups"] for h in houses])
        paired = deltas[draws].mean(axis=1)
        out["jepa_minus_"+comparison] = dict(house_mean=float(deltas.mean()),
            interval95=np.quantile(paired,[.025,.975]).tolist(), house_values=deltas.tolist())
    return out


def run(split):
    torch.set_num_threads(2)
    os.umask(0o077)
    gate = MixedPathChecker.load(LEGACY/"private_checker_mixed/checker.joblib")
    horizon = HorizonCameraPredictor.from_checkpoints(PRE/"private_train","cuda" if torch.cuda.is_available() else "cpu")
    scorer = RepairScorer(LEGACY/"private_repair/with_hint.pt")
    totals = defaultdict(lambda:defaultdict(Counter))
    houses = defaultdict(lambda:defaultdict(Counter))
    causes = defaultdict(Counter)
    selected = []
    pairing = defaultdict(Counter)
    raw = ROOT/"private_data"
    manifest=json.loads((ROOT/"view_design.json").read_text()) if (ROOT/"view_design.json").exists() else {}
    files = sorted(raw.glob(split+"_*.npz"))
    assert files
    for file in files:
        house = file.stem.split("_",1)[1]
        with np.load(file) as f, np.load(ROOT/"private_predictions"/file.name) as g:
            d = {key:f[key] for key in ("kind","group","action","plan_id","view_id","context","pose","map","reach","pixels",
                    "first_blocked","scan_cost","budget","source_index","start_pixels","scan_pixels","category")}
            p = {key:g[key] for key in g.files}
        for key in ("kind","view_id","plan_id","group"):
            assert np.array_equal(p[key],d[key]),(file,key)
        for group in np.unique(d["group"]):
            rows=np.flatnonzero(d["group"]==group)
            dd={k:v[rows] for k,v in d.items()}
            assert dd["scan_cost"].min()==dd["scan_cost"].max()==2 and dd["budget"].min()==dd["budget"].max()
            pred={k:v[rows] for k,v in p.items() if k.startswith("p_visible")}
            src_idx=int(dd["source_index"][0])
            render=ROOT/("private_render_blind60" if split=="blind" else "private_render")
            with np.load(render/file.stem/f"{src_idx}_sensor.npz") as sensor:
                choices,ok,trigger,old,lens=infer((gate,horizon,scorer),dd,pred,sensor)
                scanned=int(dd["scan_pixels"][0])
            reach=dd["reach"].astype(bool); clear=reach & (dd["pixels"]>=369)
            amount=dd["first_blocked"]
            steps=np.where(amount>0,amount,lens)
            cap=int(dd["budget"][0]); scan_cost=int(dd["scan_cost"][0])
            assert np.all((steps>0)&(steps<=lens))
            alt=(dd["kind"].astype(str)=="alt")&(lens<=cap-scan_cost)
            all_blocked=not reach[old].any()
            start=int(dd["start_pixels"][0])
            novel=(start<369 and 0<=scanned<369)
            base=choices["old"]
            was_clear=base is not None and clear[base]
            groups=["all","true_all_blocked" if all_blocked else "at_least_one_old_reachable",
                    "all_old_predicted_rejected" if trigger else "unchanged",
                    "start_unclear" if start<369 else "start_clear"]
            if scanned>=0:
                groups+=["both_start_scan_unclear" if novel else "start_or_scan_clear"]
            if all_blocked and novel:groups+=["true_all_blocked_and_start_scan_unclear"]
            if trigger:groups+=["correct_rejection" if all_blocked else "false_rejection"]
            if base is None and not trigger:groups+=["old_bounded_abstained_but_seven_has_admission"]
            if all_blocked and base is not None:groups+=["true_all_blocked_old_false_admission"]
            if was_clear:groups+=["old_already_succeeded"]
            def tally(out):
                if out is None:
                    row=Counter(groups=1,defer=1,actions=scan_cost)
                else:
                    assert lens[out]+scan_cost<=cap
                    row=Counter(groups=1,issue=1,reach=int(reach[out]),clear=int(clear[out]),actions=int(steps[out])+scan_cost,
                                collision=int(amount[out]>0), reach_not_clear=int(reach[out] and not clear[out]))
                row["additional_actions"]=row["actions"]-scan_cost
                row["ever_clear"]=int(start>=369 or scanned>=369 or row["clear"])
                row["rescued"]=int(not was_clear and row["clear"])
                row["regressed"]=int(was_clear and not row["clear"])
                return row
            for arm,chosen in choices.items():
                row=tally(chosen)
                for subset in groups:totals[subset][arm].update(row)
                houses[house][arm].update(row)
            # Explicit controls: bench success requires an actual viewpoint or a confirmed current image.
            for arm, seen in (("stay_at_source",start),("shared_sweep_best",max(start,scanned))):
                row=Counter(groups=1,clear=int(seen>=369),actions=0 if arm=="stay_at_source" else scan_cost)
                for subset in groups:totals[subset][arm].update(row)
                houses[house][arm].update(row)
            winner=choices["jepa"]
            for other,chosen in choices.items():
                mine=winner is not None and clear[winner]; theirs=chosen is not None and clear[chosen]
                pairing[other].update(win=int(mine and not theirs),loss=int(theirs and not mine))
            if winner is not None and clear[winner]:cause="reached_clear"
            elif not trigger: cause=("unchanged_old_issued_obstructed" if base is not None and not reach[base]
                    else "unchanged_old_issued_unclear" if base is not None else "bounded_old_abstention_despite_other_admitted")
            elif not (reach&alt).any():cause="no_physically_reachable_alternative"
            elif not (clear&alt).any():cause="alternatives_reachable_but_none_clear"
            elif not (ok&clear&alt).any():cause="all_useful_alternatives_vetoed"
            elif not any(clear[i] for i in shortest_per_view(dd["view_id"],ok&alt,lens)):
                cause="shortest_admitted_route_blocked_despite_sibling"
            elif winner is None:cause="declined_with_viable_admitted_alternative"
            elif not reach[winner]:cause="selected_route_blocked_with_viable_alternative"
            else:cause="selected_view_unclear_with_viable_alternative"
            for subset in groups:
                causes[subset][cause]+=1
                oracle=totals[subset]["candidate_ceiling"]
                oracle.update(groups=1,old_reach=int(reach[old].any()),old_clear=int(clear[old].any()),
                              alt_reach=int((reach&alt).any()),alt_clear=int((clear&alt).any()),
                              alt_admitted_clear=int((ok&clear&alt).any()))
            selected.append(dict(house=house,group=str(group),source=src_idx,scan_pixels=scanned,
                                 old=int(rows[base]) if base is not None else None,
                                 jepa=int(rows[winner]) if winner is not None else None,
                                 trigger=trigger,reason=cause))
        print("Evaluated",file.stem,flush=True)
    if manifest.get("cohorts",{}).get(split):assert len(houses)==len(manifest["cohorts"][split])
    assert totals["all"]["jepa"]["regressed"]==0
    assert sum(causes["all"].values())==totals["all"]["jepa"]["groups"]
    result=dict(houses=len(houses),subsets=totals,failure_decomposition=causes,paired_changes=pairing,
                paired_clear=interval(houses, next(iter(houses.values())).keys()),
                homes=[dict(rows=houses[k]) for k in sorted(houses)],
                protocol="Frozen original first-3 plus bounded repairs. Fallback only on old abstention and all seven source-depth rejections. Same 2 scan actions and per-group cap. Same-view shortest admitted representative. Success is measured only on execution with at least 369 same-instance pixels. Ever-clear and shared-scan/current controls are separate. First observed contact stops one local attempt.")
    (ROOT/f"{split}_evaluation.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    private=ROOT/"private_evaluation";private.mkdir(exist_ok=True)
    (private/f"{split}_choices.json").write_text(json.dumps(selected)+"\n")
    print(json.dumps(totals["all"]),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("split",choices=("val","blind"));args=ap.parse_args()
    with threadpool_limits(limits=2):run(args.split)
