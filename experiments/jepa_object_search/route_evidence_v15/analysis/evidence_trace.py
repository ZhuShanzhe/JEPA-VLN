"""Aggregate the two accepted decisions, observation opportunities, and reporting gaps."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_case(pilot, row, certified):
    eid = row["episode_id"]
    receipt = read(pilot/"comparison/accepted/second"/(eid+".json"))
    result = Path(receipt["result"])
    replay = pilot/"replay"/(eid+".json")
    assert sha(result) == certified["result_sha256"] == receipt["result_sha256"]
    assert sha(replay) == certified["replay_sha256"]
    events_path = result.parent/"agent/v13_routes.jsonl"
    events = [json.loads(line) for line in events_path.read_bytes().splitlines()]
    checks = [r for r in events if r["event"] == "eligibility"]
    eligible = [r for r in checks if r["reason"] is None]
    bypass = Counter(r["reason"] for r in checks if r["reason"] is not None)
    starts = [r for r in events if r["event"] == "scan_start"]
    scans = [r for r in events if r["event"] == "scan_end"]
    ranks = [r for r in events if r["event"] == "rank"]
    assert len(eligible) == len(starts)
    assert len(ranks) == len(certified["ranked_decisions"])
    bins = Counter()
    evidence_changed = exploration_changed = in_both = 0
    shadow_changes = native_changes = verified_v13 = verified_native = 0
    audit = []
    for r, match in zip(ranks, certified["ranked_decisions"]):
        assert r["step"] == match["step"]
        i = r["index"]
        assert i == max(range(3), key=r["final_scores"].__getitem__)
        pred = r["evidence_probability"]
        evidence = r["evidence_bonus"]
        bonus = r["exploration_bonus"]
        base = r["v13_scores"]
        assert all(abs(r["final_scores"][k]-base[k]-evidence[k]-bonus[k])<1e-6 for k in range(3))
        assert r["no_exploration_index"] == max(range(3), key=lambda k:base[k]+evidence[k])
        assert r["no_evidence_index"] == max(range(3), key=lambda k:base[k]+r["adaptive_weight"]*.35*r["unvisited_fraction"][k])
        d = r["clue_deficit"]
        bins["weak_or_absent" if d >= .75 else "strong" if d <= .25 else "intermediate"] += 1
        changes_explore = i != r["no_exploration_index"]
        changes_predict = i != r["no_evidence_index"]
        changes_old = i != r["v13_index"]
        assert changes_old == match["change_from_v13"] and changes_explore == match["change_with_exploration"]
        exploration_changed += changes_explore
        evidence_changed += changes_predict
        in_both += changes_explore and changes_predict
        shadow_changes += changes_old
        native_changes += i != 0
        native_physical = (match["native_execution"] or {}).get("verified", False)
        old_physical = (match["v13_execution"] or {}).get("verified", False)
        verified_v13 += old_physical
        verified_native += native_physical
        audit.append(dict(step=r["step"], deficit=round(d, 3),
                          explore_weight=round(r["adaptive_weight"], 3),
                          selected_new_evidence=round(pred[i], 3),
                          candidate_evidence_spread=round(max(pred)-min(pred), 3),
                          candidate_exploration_bonus_spread=round(max(bonus)-min(bonus), 3),
                          changed_by_exploration=changes_explore, changed_by_evidence=changes_predict,
                          changed_vs_v13=changes_old, executed_vs_v13=old_physical,
                          changed_vs_native=i != 0, executed_vs_native=native_physical))
    metrics = certified["result"]
    keys = ("steps", "quota", "found", "capped_found", "approached", "clear_exposure",
            "clear_and_approached", "clear_never_approached", "clear_unreported",
            "approached_unreported", "required_missing_never_approached",
            "required_missing_approached_unreported", "reports", "reports_yielding_new_instances",
            "repeat_only_reports", "no_target_in_range_reports")
    return dict(ordinal=row["ordinal"], eligibility_checks=len(checks), eligible=len(eligible),
                first_eligible_step=eligible[0]["step"] if eligible else None,
                bypassed=sum(bypass.values()), bypass_reasons=dict(bypass),
                scan_starts=len(starts), scan_end_reasons=dict(Counter(r["reason"] for r in scans)),
                scan_confirmed_turns=sum(r["physical_turns"] for r in scans),
                ranks=len(ranks), first_rank_step=ranks[0]["step"] if ranks else None,
                deficit_bins=dict(bins), exploration_changed_choice=exploration_changed,
                evidence_changed_choice=evidence_changed, both_changed_choice=in_both,
                changed_vs_v13=shadow_changes, confirmed_vs_v13=verified_v13,
                changed_vs_native=native_changes, confirmed_vs_native=verified_native,
                rank_summary=audit, search_and_reporting={k:metrics[k] for k in keys},
                commit_ends=certified["commit_ends"], trace_sha256=sha(events_path))


def main(project):
    kit = project/"jepa_evidence_v15"
    phase = kit/"panel/phase1_decision.json"
    pilot = project/"mapping_runtime/pilots/jepa_evidence_v15_first2"
    decision = read(phase)
    rows = read(pilot/"panel.json")["rows"]
    assert [r["ordinal"] for r in rows] == [r["ordinal"] for r in decision["cases"]] == [1,14]
    assert sha(pilot/"five_comparison.json") == decision["first_summary_sha256"]
    cases = [one_case(pilot, row, cert) for row,cert in zip(rows,decision["cases"])]
    result = dict(cases=cases, phase1_sha256=sha(phase),
                  deficit_bins="weak or absent >=0.75; strong <=0.25; intermediate otherwise; scored states only",
                  clear_exposure="103 target semantic pixels on two successive distinct camera views, 320x320",
                  cue_source="Episode-local classifier evidence is distinct from the evaluator's clear instance exposure.",
                  decision_scope="Ablations rescore identical candidate inputs; confirmed_vs counts rely on independent physical fork replay.")
    output = kit/"analysis/trace_summary.json"
    document = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if output.exists():
        assert output.read_text() == document
    else:
        with output.open("x") as f:
            f.write(document)
    print(json.dumps(cases, ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
