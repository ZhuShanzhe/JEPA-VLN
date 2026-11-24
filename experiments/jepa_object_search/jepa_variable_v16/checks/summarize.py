"""Export only aggregate replay statistics, never episode geometry or action traces."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_bytes())


def summarize(cases):
    ranked = [c for c in cases if "selected" in c]
    changed = [c for c in ranked if c["status"] == "first_difference_observed"]
    moved = [c for c in changed if c["separation_m"] > .005]
    turned = [c for c in changed if c["separation_m"] <= .005 and c["separation_deg"] > 1.]
    sizes = Counter(len(c["lengths"]) for c in ranked)
    lengths = [length for c in ranked for length in c["lengths"]]
    return dict(activation_count=len(cases), ranked=len(ranked), ranked_candidate_count=dict(sorted(sizes.items())),
                delayed_watch_starts=sum(bool(c.get("delayed_watch_started")) for c in cases),
                minimum_scored_length=min(lengths), maximum_scored_length=max(lengths),
                mixed_length_decisions=sum(len(set(c["lengths"]))>1 for c in ranked),
                keep_native=len([c for c in ranked if c["status"]=="kept_native"]),
                fallback_without_scan=dict(Counter(c["reason"] for c in cases if c["status"]=="native_fallback")),
                committed_changes=len(changed), position_different=len(moved), heading_only_different=len(turned),
                no_measured_pose_difference=len(changed)-len(moved)-len(turned),
                safety_interruptions=dict(Counter(c["status"] for c in ranked if c["status"] not in
                                                  ("kept_native","first_difference_observed"))))


folder=Path(sys.argv[1])
natural, controls, safety = [read(folder/s/"receipt.json") for s in ("junctions","controls","safety")]
assert all(x["passed"] for x in (natural,controls,safety))
n=[c for c in natural["results"] if not c["forced"]]
assert len(n)==12 and len(controls["results"])==2
result = {"protocol":"independent local HM3D interventions at frozen V15 eligibility events",
          "comparison_endpoint":"first distinct primitive, or first original action if unchanged",
          "position_threshold_m":.005,"heading_threshold_deg":1.,
          "upstream":"archived MOST candidate geometry and actions; Habitat-rendered causal RGB",
          "network":"IPv4 and IPv6 sockets blocked in evaluator and scorer subprocess; zero paid calls",
          "natural":summarize(n), "natural_per_task":{
              label:summarize([c for c in n if c["ordinal"]==i]) for i,label in ((1,"clock"),(14,"wall_art"))},
          "forced_last_option_controls":summarize(controls["results"]),
          "public_dispatcher_checks":{c["case"]:c["end"] for c in safety["cases"]}}
if (folder/"sources.sha256").exists():
    result["verified_source_manifest_sha256"]=hashlib.sha256((folder/"sources.sha256").read_bytes()).hexdigest()
with (folder/"public_summary.json").open("x") as fp:
    json.dump(result,fp,indent=2,allow_nan=False)
    fp.write("\n")
print(json.dumps(result,ensure_ascii=False))
