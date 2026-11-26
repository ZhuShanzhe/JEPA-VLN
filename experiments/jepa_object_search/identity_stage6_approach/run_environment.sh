#!/usr/bin/env bash
set -eu
approach_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
approach_project="$(dirname "$approach_dir")"
source "$approach_project/jepa_box_v26/adapter/environment.sh"
export PYTHONPATH="$approach_dir:$approach_project/identity_stage5_confirmation:$approach_project/identity_stage4_geometry/provider:$approach_project/identity_stage3_model/integration:$PYTHONPATH"
export NAV_STAGE3_IDENTITY=jepa NAV_STAGE4_GEOMETRY=keyframes_only NAV_BOX_PROVIDER=dino
export MOST_FROZEN_AGENT=approach_agent:ApproachAgent
export NAV_LOCAL_BOXES=1 NAV_COVISIBLE_GUARD=1
export MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$approach_dir"
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [[ "${1:-}" == "prepare_evaluation.py" ]]; then
        exec "${NAV_SUPERVISOR_PYTHON:-$HOME/miniconda3/bin/python}" -Bu "$@"
    fi
    exec "$MOST_EVAL_PYTHON" -Bu "$@"
fi
