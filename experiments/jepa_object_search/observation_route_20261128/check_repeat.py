"""Use private receipt hashes to verify both controlled runs against the released integration."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("initial_attempt",type=Path)
    parser.add_argument("final_attempt",type=Path)
    args = parser.parse_args()
    initial = json.loads((HERE/"loop_results.json").read_text())
    final = json.loads((HERE/"loop_final.json").read_text())
    assert initial["results"] == final["results"]
    sources = final["source_sha256"].copy()
    assert all(digest(HERE/name)==value for name,value in sources.items())
    for name in ("terminal_prefix.py","host.py","probe_branches.py","probe_score.py","review_loop.py"):
        sources[name] = digest(HERE/name)
    n = 0
    for i in (0,1):
        for arm in ("regional","combined"):
            for ext in ("poses","events"):
                name = f"{i}_{arm}.{ext}"
                assert digest(args.initial_attempt/name)==digest(args.final_attempt/name),name
                n += 1
    receipt = dict(identical_private_pose_and_event_files=n,matched_arms=4,
                   physical_actions_per_pass=1200,same_public_metrics=True,
                   identical_replay_is_not_an_independent_sample=True,
                   final_source_sha256=sources)
    (HERE/"loop_repeat.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("Matched four private trajectories and four decision streams across the original and final passes.")
