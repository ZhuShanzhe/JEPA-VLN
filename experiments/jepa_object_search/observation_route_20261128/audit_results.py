"""Reconcile public counts, execution receipts and per-arm denominators."""
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT/name).read_text(encoding="utf8"))


def main():
    loop, visibility, probes = map(load, ("loop_results.json", "loop_visibility.json", "probe_results.json"))
    assert loop["paid_calls"] == probes["paid_calls"] == 0
    assert len(loop["results"]) == len(visibility["results"]) == 4
    firsts = {(r["house_index"],r["arm"]):r for r in visibility["results"]}
    assert set(firsts) == {(i,a) for i in (0,1) for a in ("combined","regional")}
    for r in loop["results"]:
        v = firsts[r["house_index"],r["arm"]]
        assert r["steps"] == 300 and v["frames"] == 301
        assert v["goal"] == r["goal"] and v["close_and_visible"] <= min(r["distinct_approached"],v["visible"])
        assert sum(v["first_visible_phase"].values()) == v["visible"]
        assert sum(v["first_close_and_visible_phase"].values()) == v["close_and_visible"]
    rows = probes["rows"]
    assert len(rows) == 32 and {(r["house"],r["sample"]) for r in rows if r["mode"]=="discover"} == {
        (i,t) for i in range(1,9) for t in range(3)}
    groups = dict(discovery=[r for r in rows if r["mode"]=="discover"],
                  source_verified=[r for r in rows if r["inspection_valid"] is True],
                  source_incorrect=[r for r in rows if r["inspection_valid"] is False])
    assert [len(x) for x in groups.values()] == [24,4,4]
    for k,group in groups.items():
        summary = probes["summary"][k]
        assert summary["candidate_routes"]==sum(r["choices"] for r in group)
        assert summary["physical_actions"]==sum(r["total_candidate_actions"] for r in group)
        assert summary["gate_counts"]==dict(Counter(r["gate"] for r in group))
        for ranker in ("model","geometry","regional"):
            assert summary[ranker]["new_instances"]==sum(r[ranker]["new_instances"] for r in group)
            for r in group:
                assert r[ranker]["new_instances"]<=r["best_new_instances"]
                assert r[ranker]["close_and_visible"]<=r["best_close_visible"]
                assert 1<=r[ranker]["length"]<=r[ranker]["planned"]<=4
    print("Verified 4 paired-arm house runs, 1,204 evaluator frames and 32 independently scored branch points.")


if __name__=="__main__":
    main()
