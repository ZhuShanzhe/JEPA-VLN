"""Offline analysis of accepted MOST-Nav ManyON runs. Never imported by a policy."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile


ROOT = Path(os.environ.get("NAV_ROOT", "/root/autodl-tmp/jepa_object_search"))
SCENES = ROOT / "frontier_scenes"
sys.path.insert(0, str(SCENES / "most_release"))
sys.path.insert(0, str(ROOT / "mapping_runtime/batch"))
import paired
from evaluation.main.evaluator import _target_approach_distance
import habitat_sim


METRICS = ("sr", "osr", "precision", "recall", "f1", "spl_multi")
PARTS = ("success", "insufficient_approach_quota", "sufficient_approach_but_underreported",
         "diagnostic_unknown")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# Habitat's Python 3.9 predates hashlib.file_digest used by the batch Python 3.12.
paired.digest = digest


def check(test, message):
    if not test:
        raise ValueError(message)


def atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        json.dump(obj, out, ensure_ascii=False, allow_nan=False, indent=2)
        out.write("\n")
        tmp = out.name
    os.replace(tmp, path)


def approach_steps(e, approached):
    """Map official reportable IDs to their first observed trajectory step on the same navmesh."""
    if not approached:
        return {}
    mesh = Path(e["scene_id"]).resolve().with_suffix(".navmesh")
    check((SCENES / "hm3d_v02/val").resolve() in mesh.parents, "unexpected scene path")
    pf = habitat_sim.PathFinder()
    check(pf.load_nav_mesh(str(mesh)), "navmesh loading failed")
    d = e["paper"]["episode_definition"]
    views = {int(k): v for k, v in d["view_points"].items()}
    positions = {int(k): v for k, v in d["target_positions"].items()}
    threshold = e["paper"]["matching"]["report_threshold_m"]
    check(math.isclose(threshold, 0.25), "report threshold drift")
    seen, result = set(), {}
    for entry in sorted(e["trajectory_positions"], key=lambda x: x["step"]):
        pos = tuple(entry["position"])
        if pos in seen:
            continue
        seen.add(pos)
        for oid in approached - result.keys():
            distance = _target_approach_distance(pf, pos, oid, positions, views)
            if distance is not None and distance <= threshold:
                result[oid] = int(entry["step"])
        if len(result) == len(approached):
            break
    return result


def decision_diagnostics(e, first, geometry_known):
    trace = e["paper"]["agent_trace"]
    decisions = sorted(trace["decisions"], key=lambda v: v["step"])
    steers = {v["decision_id"]: v for v in trace["steering_events"]}
    counts = Counter()
    for idx, d in enumerate(decisions):
        if d["decision_origin"] != "vlm":
            counts["non_vlm_decisions"] += 1
            continue
        counts["vlm_decisions"] += 1
        if d["validation"] != "ok":
            counts["invalid_vlm_decisions"] += 1
            continue
        candidates = {r["id"]: r for r in d["world_state"].get("frontiers", [])}
        counts["validated_decisions"] += 1
        counts["validated_decisions_without_frontier"] += len(candidates) == 0
        counts["validated_decisions_with_multiple_frontiers"] += len(candidates) >= 2
        if d["output"]["action"] != "GOTO_FRONTIER":
            continue
        counts["requested"] += 1
        selected = d["output"].get("target_id")
        s = steers.get(d["decision_id"])
        if s is None:
            counts["unmatched_steering"] += 1
            continue
        if len(candidates) >= 2 and selected in candidates:
            # The selector runs before the follower tries the original path.
            counts["selector_applicable_requests"] += 1
        if not s["applied"]:
            counts["blocked"] += 1
            continue
        check(selected in candidates and s["requested"]["target_id"] == selected, "trace selection mismatch")
        counts["applied"] += 1
        if len(candidates) < 2:
            counts["applied_without_alternative"] += 1
            continue
        counts["applied_with_alternative"] += 1
        counts["selected_recently_attempted"] += bool(candidates[selected].get("recently_attempted"))
        if not geometry_known:
            continue
        start = d["step"]
        end = decisions[idx + 1]["step"] if idx + 1 < len(decisions) else e["steps_taken"]
        # Observation at start precedes that decision. Boundary observation belongs to the preceding window.
        new_ids = sum(start < step <= end for step in first.values())
        counts["windows_with_geometry"] += 1
        counts["windows_with_new_approach"] += new_ids > 0
        counts["windows_without_new_approach"] += new_ids == 0
        if sum(step <= start for step in first.values()) < e["required_target_count"]:
            counts["windows_before_approach_quota"] += 1
            counts["prequota_windows_with_new_approach"] += new_ids > 0
            counts["prequota_windows_without_new_approach"] += new_ids == 0
    counts["episodes_with_selector_applicable_request"] = int(counts["selector_applicable_requests"] > 0)
    counts["episodes_with_applied_alternative"] = int(counts["applied_with_alternative"] > 0)
    counts["episodes_with_prequota_alternative"] = int(counts["windows_before_approach_quota"] > 0)
    return dict(counts)


def episode_summary(e, condition):
    k = e["required_target_count"]
    found = set(e["reached_targets"])
    approached = set(e["approached_target_ids"])
    check(k > 0 and len(found) == e["found_count"] and e["target_mode"] == "many" and
          bool(e["sr"]) == (len(found) >= k), "inconsistent ManyON quota")
    # A missing individual distance on an inaccessible or uncomputable object
    # cannot prove inadequate geometric coverage when fewer than K were reached.
    ds = e.get("target_min_distances", {})
    all_distances = (len(ds) == e["total_targets"] and
                     all(v is not None and math.isfinite(v) for v in ds.values()))
    approach_known = (e["osr"] is not None and found <= approached and
                      bool(e["osr"]) == (len(approached) >= k) and
                      (all_distances or len(approached) >= k))
    if e["sr"]:
        part = "success"
    elif not approach_known:
        part = "diagnostic_unknown"
    elif len(approached) < k:
        part = "insufficient_approach_quota"
    else:
        part = "sufficient_approach_but_underreported"
    missing = max(0, k - len(found))
    shortfall = dict(total_missing=missing)
    if approach_known:
        gap = max(0, k - len(approached))
        shortfall.update(missing_to_reach_quota=gap, reached_but_unreported=missing - gap)
    else:
        shortfall["unresolved"] = missing
    c = Counter()
    for r in e["report_events"]:
        c["actions"] += 1
        c["no_eligible_target_in_range"] += not r["near_targets"]
        c["with_new_target"] += bool(r["new_targets"])
        c["with_multiple_new_targets"] += len(r["new_targets"]) > 1
        c["with_over_quota_claim"] += bool(r["over_quota_targets"])
        c["mixed_new_and_over_quota"] += bool(r["over_quota_targets"] and r["new_targets"])
        c["matched_without_new_or_over_quota"] += bool(r["near_targets"] and
                                                       not r["new_targets"] and not r["over_quota_targets"])
        c["new_distinct_instances"] += len(r["new_targets"])
        c["over_quota_claims"] += len(r["over_quota_targets"])
    c["new_instance_credits"] = min(k, c["new_distinct_instances"])
    check(c["new_distinct_instances"] == len(found) and
          c["new_instance_credits"] == e["true_positive_count"] and
          c["over_quota_claims"] + c["no_eligible_target_in_range"] == e["false_positive_count"],
          "official report counts disagree")
    c["episodes_without_report_action"] = int(not c["actions"])
    c["official_repeated_claims"] = e["repeat_report_count"]
    c["official_target_claims"] = e["target_report_count"]
    first = approach_steps(e, approached) if approach_known else {}
    replay_complete = bool(approach_known and set(first) == approached)
    counts = decision_diagnostics(e, first, replay_complete)
    counts["episodes_with_replay_geometry"] = int(replay_complete)
    pool = {int(v) for v in e.get("pool_matched_target_ids", [])}
    pool_summary = {"episodes_pool_available": int(e["pool_tse_status"] == "agent_reported_pool")}
    if e["pool_tse_status"] == "agent_reported_pool" and approach_known:
        missed = approached - found
        pool_summary.update(approached_unreported_matching_pool=len(missed & pool),
                            approached_unreported_without_pool_match=len(missed - pool))
    return dict(scene=e["scene_id"], condition=condition, quota=k, part=part,
                shortfall=shortfall, reports=dict(c), frontiers=counts, pool=pool_summary,
                termination=e["failure_reason"] or "success", metrics={x:e[x] for x in METRICS})


def collapse(rows, key):
    total = Counter()
    for row in rows:
        total.update(row[key])
    return dict(total)


def interval(rows):
    scenes = defaultdict(list)
    for row in rows:
        scenes[row["scene"]].append(row["metrics"])
    if len(scenes) < 2:
        return None
    # Uniformly resample scenes and carry all completed episodes from selected scenes.
    sample = {}
    for metric in ("sr", "osr", "recall", "f1"):
        contrib = []
        for data in scenes.values():
            vals = [r[metric] for r in data if r[metric] is not None]
            contrib.append((sum(vals), len(vals)))
        rng = random.Random(4471)
        estimates = []
        for _ in range(2000):
            total = count = 0
            for item in rng.choices(contrib, k=len(contrib)):
                total += item[0]
                count += item[1]
            if count:
                estimates.append(total / count)
        if estimates:
            estimates.sort()
            sample[metric] = [estimates[round(p * (len(estimates)-1))] for p in (0.025,0.975)]
    return sample


def group(rows):
    counts = Counter(r["part"] for r in rows)
    means = {}
    for metric in METRICS:
        vs = [r["metrics"][metric] for r in rows if r["metrics"][metric] is not None]
        means[metric] = statistics.mean(vs) if vs else None
    frontier_by_failure = {key: collapse([r for r in rows if r["part"] == key], "frontiers")
                           for key in PARTS}
    return {"n": len(rows), "scene_count": len({r["scene"] for r in rows}), "mean": means,
            "failure_partition": {p:counts[p] for p in PARTS}, "quota_shortfall": collapse(rows, "shortfall"),
            "termination": dict(Counter(r["termination"] for r in rows)), "report_events": collapse(rows,"reports"),
            "frontiers": collapse(rows, "frontiers"), "frontiers_by_failure_partition": frontier_by_failure,
            "pool_proxy": collapse(rows, "pool"), "ci95": interval(rows),
            "by_quota": {str(k): {"n": len(group_rows), "sr": statistics.mean(v["metrics"]["sr"] for v in group_rows),
                                  "failure_partition": dict(Counter(v["part"] for v in group_rows))}
                         for k in sorted(set(r["quota"] for r in rows))
                         for group_rows in [[r for r in rows if r["quota"] == k]]}}


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--campaign", type=Path)
    inputs.add_argument("--development-result", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    summaries = []
    if args.campaign:
        campaign = args.campaign.resolve()
        lock = paired.data(campaign / "lock.json")
        panel = paired.panel()
        check(lock["episodes"] == panel and lock["namespace"] == "formal_frozen" and lock["arms"] == ["native"],
              "formal lock/panel mismatch")
        all_ids = {r["episode_id"] for r in panel}
        refs = list((campaign / "accepted/native").glob("*.json"))
        check(all(p.stem in all_ids for p in refs), "unexpected accepted episode")
        accepted_ids = {p.stem for p in refs}
        if args.require_complete:
            check(accepted_ids == all_ids, "full 300-episode acceptance is required")
        cache_id = hashlib.sha256(Path(__file__).read_bytes() + (campaign / "lock.json").read_bytes()).hexdigest()
        for r in panel:
            eid = r["episode_id"]
            if eid not in accepted_ids:
                continue
            e = paired.accepted(campaign, lock, "native", r)
            ref = read(paired.accepted_path(campaign, "native", eid))
            cache = campaign / "analysis/private_cache" / cache_id / (ref["result_sha256"] + ".json")
            if not cache.exists():
                atomic(cache, episode_summary(e, r["condition"]))
            summaries.append(read(cache))
        expected = len(panel)
        cohort = "formal_accepted_native_only"
    else:
        check(not args.require_complete, "development input cannot satisfy formal completeness")
        e, = read(args.development_result)["episodes"]
        check(e["episode_id"] not in {r["episode_id"] for r in paired.panel()}, "frozen ID in development mode")
        summaries = [episode_summary(e, "development")]
        expected, cohort = 1, "non_frozen_engineering_only"
    groups = {"all": group(summaries)}
    for condition in ("many_category", "many_attribute"):
        groups[condition] = group([r for r in summaries if r["condition"] == condition])
    report = {"schema_version":1, "cohort":cohort, "expected": expected, "accepted": len(summaries),
              "complete": bool(args.campaign and expected == len(summaries)), "groups": groups,
              "interpretation": [
                  "Geometry uses official <=0.25m reporting criterion; it does not establish pixel visibility or recognition.",
                  "The additive shortfall has unit missing quota credits. A single report may credit more than one overlapping target.",
                  "A pool match is a nearest-centroid spatial proxy within 2m; it is not object identity or correct semantic recognition.",
                  "Frontier windows cover observed movement until the next decision with geometry available. Only newly approached distinct targets count. Revisit and exploration without immediate reward can be useful. Alternative outcomes are unobserved.",
                  "Success, oracle success, precision, recall, F1 and SPL are per-episode means over defined values. CI resamples scenes with all their accepted episodes, 2000 draws. Incomplete traversal is not an unbiased sample of the final panel."]}
    atomic(args.out, report)
    print(f"{cohort}: {len(summaries)}/{expected}; complete={report['complete']}")


if __name__ == "__main__":
    run()
