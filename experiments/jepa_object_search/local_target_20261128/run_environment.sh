#!/usr/bin/env bash
set -eu
local_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$local_root")/approach_search_20261128/run_environment.sh"
export PYTHONPATH="$local_root:$local_root/waypoints:$PYTHONPATH"
export MOST_FROZEN_AGENT="local_agent:LocalApproachAgent"
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    exec "$MOST_EVAL_PYTHON" -Bu "$@"
fi
