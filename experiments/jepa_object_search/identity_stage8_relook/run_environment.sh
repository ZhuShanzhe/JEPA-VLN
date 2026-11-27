#!/usr/bin/env bash
set -eu
relook_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$relook_dir")/identity_stage7_transfer/run_environment.sh"
export PYTHONPATH="$relook_dir:$PYTHONPATH"
export MOST_FROZEN_AGENT="weak_route_agent:WeakRouteAgent"
cd "$relook_dir"
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    exec "$MOST_EVAL_PYTHON" -Bu "$@"
fi
