"""Post-episode, assessor-only visibility check on the delivered clock trajectory."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ["NAV_ROOT"])
RT = ROOT / "mapping_runtime"
STUDY = RT / "pilots" / "identity_stage8_weak_relook"


def load(path):
    return json.loads(path.read_bytes())


def describe(replay):
    names = ("any", "moderate", "large")
    counts = ["visible_steps", "first_qualifying_step"]
    return {
        "replayed_steps": replay["steps_replayed"],
        "official_credits": replay["reported_distinct"],
        "distinct_visible": replay["count_exposed"],
        "max_replay_position_error_m": replay["max_position_error_m"],
        "reference_rgb_checks": replay["reference_rgb_frames"],
        "maximum_rgb_mae_64": replay["max_reference_rgb_mae_at_64"],
        "unconfirmed_annotation_targets": replay["never_seen_unconfirmed"],
        "per_target_sorted_by_peak_pixels": [
            dict(peak_pixels=t["max_pixels"], entered_range=t["approached"],
                 **{n: {k: t["exposure"][n][k] for k in counts} for n in names})
            for t in sorted(replay["targets"], key=lambda t: -t["max_pixels"])
        ],
    }


def main(study):
    from verify_rank import deny_ip_sockets
    deny_ip_sockets()
    sys.path.insert(0, str(RT / "analysis" / "visibility"))
    from visibility_replay import replay
    row, = load(study / "panel.json")["rows"]
    name = row["episode_id"] + ".json"
    orig = load(RT / "analysis/visibility/native_20261115/verified" / name)
    assert orig["source_result_sha256"] == row["result_sha256"]
    accepted = load(study / "comparison/accepted/second" / name)
    result = Path(accepted["result"])
    checksum = hashlib.sha256(result.read_bytes()).hexdigest()
    assert checksum == accepted["result_sha256"]
    service = RT / "runs" / load(study / "mapper_process.json")["run"]
    private = study / "private_visibility" / name
    if not private.exists():
        replay(result, service, private)
    current = load(private)
    assert current["source_result_sha256"] == checksum
    assert current["episode_id"] == orig["episode_id"] == row["episode_id"]
    assert current["thresholds"] == orig["thresholds"]
    output = dict(scope="Post-episode assessor only, identical Habitat task. No policy feedback.",
                  resolution=current["semantic_resolution"], thresholds=current["thresholds"],
                  native=describe(orig), revised=describe(current), revised_result_sha256=checksum)
    dest = Path(__file__).resolve().with_name("visibility_summary.json")
    if dest.exists():
        assert load(dest) == output
    else:
        dest.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["revised"]["distinct_visible"]))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv)>1 else STUDY)
