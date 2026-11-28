#!/usr/bin/env bash
set -eu
inspect_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$inspect_root")/gpt_region_20261128/run_environment.sh"
export PYTHONPATH="$inspect_root:$PYTHONPATH"
export MOST_FROZEN_AGENT="inspect_agent:InspectOnlyAgent"
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    exec "$MOST_EVAL_PYTHON" -Bu "$@"
fi
