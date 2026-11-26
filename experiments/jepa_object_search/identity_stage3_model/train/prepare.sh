#!/usr/bin/env bash
# Arguments: project data root, compatible HSSD assets, Habitat Python, detector/encoder Python.
set -euo pipefail
project=$(realpath "$1")
assets=$(realpath "$2")
sim=$(realpath -s "$3")
model=$(realpath -s "$4")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
private="$project/identity_stage3_model/train/private"
mkdir -p "$private"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet
umask 077
for split in train val; do
    if [[ ! -f "$private/${split}_clock_seeds.jsonl" ]]; then
        "$sim" "$here/clock_seeds.py" "$project" "$assets" "$private" "$split"
    fi
    for suffix in '' _clocks; do
        name="$split$suffix"
        dir="$private/$name"
        if [[ ! -f "$dir/frames.jsonl" ]]; then
            params=()
            if [[ -n "$suffix" ]]; then params=(clocks); fi
            "$sim" "$here/training_twins.py" "$project" "$assets" "$dir" "$split" "${params[@]}"
        fi
        if [[ ! -f "$dir/det/predictions.jsonl" ]]; then
            "$model" "$project/jepa_reid_v21/detector/extra_panel.py" "$project" "$dir" "$dir/det"
        fi
        "$model" "$project/jepa_reid_v21/model/encode_proposals.py" "$project" --frames "$dir/frames.jsonl" \
            --predictions "$dir/det/predictions.jsonl" --cache "$private/$name.pt"
    done
done
