"""Verify all HSSD pilot receipts, checkpoints and RGB parity against their retained files."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt",type=Path,required=True)
    args = parser.parse_args()
    audit_root = here.parent / "a_exploration" / "paired_branches_v2"
    audited = json.loads((audit_root / "audit.json").read_text())
    assert audited["complete"] and audited["houses"] == {"train":12,"dev":4}
    seeds = [17,18,19]
    results = {}
    for seed in seeds:
        directory = here / "runs" / f"hssd_v2_seed{seed}_e5"
        receipt = json.loads((directory / "receipt.json").read_text())
        assert receipt["seed"] == seed and receipt["train_dev_houses_disjoint"]
        for name,digest in receipt["implementation_hashes"].items():
            assert sha(here / name) == digest
        for name,digest in receipt["data_hashes"].items():
            assert sha(audit_root / name) == digest
        results[seed] = {}
        for mode in ("direct","predictive"):
            result = receipt["results"][mode]
            assert len(result["training"]) == 5 and result["checks"]["candidate_permutation_equivariant"]
            assert result["checks"]["illegal_action_rejected"] and result["checks"]["mirrored_plan_max_score_delta"] > 0
            filename = f"{mode}.pt"
            assert receipt["checkpoint_sha256"][filename] == sha(directory / filename)
            choices = [json.loads(row) for row in (directory / f"{mode}_choices.jsonl").read_text().splitlines()]
            dev = result["development"]
            assert dev["states"] == len({row["state"] for row in choices}) == len(choices)
            assert sum(r["selected_proxy"] for r in choices) == dev["chosen_positive"]
            assert sum(r["best_available_proxy"] for r in choices) == dev["oracle_positive_states"] == 2
            assert dev["informative_states"] == 2 and dev["comparable_pairs"] == 5
            results[seed][mode] = {"chosen_positive":dev["chosen_positive"],"states":len(choices)}
    directory = here / "runs" / "hssd_v2_seed17_e5"
    rgb = json.loads((directory / "rgb_acceptance.json").read_text())
    assert rgb["raw_context_parity"] and rgb["same_candidate_as_cached"]
    assert rgb["checkpoint_sha256"] == sha(directory / "predictive.pt")
    positives_by_house = {k:v["positive_branches"] for k,v in audited["by_house"].items() if k.startswith("dev/")}
    result = dict(verified_seeds=seeds,checked_checkpoints=6,verified_source_and_input_hashes=True,
                  train=audited["counts"]["train"],dev=audited["counts"]["dev"],
                  houses=audited["houses"],dev_positive_branches_by_house=positives_by_house,
                  choices=results,real_rgb_acceptance=rgb)
    with args.receipt.open("x") as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
