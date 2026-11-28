#!/usr/bin/env bash
set -eu
gate_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(dirname "$gate_root")"
source "$project_root/local_target_20261128/run_environment.sh"
source "$project_root/ops/profiles/apiyi_gpt4o/activate.sh"
export PYTHONPATH="$gate_root:$PYTHONPATH"
export MOST_FROZEN_AGENT="gpt_agent:GPTRegionAgent"
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    exec "$MOST_EVAL_PYTHON" -Bu "$@"
fi
