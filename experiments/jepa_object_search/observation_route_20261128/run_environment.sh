#!/usr/bin/env bash
set -eu
observation_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$observation_root")/multiview_nav_20261128/run_environment.sh"
export PYTHONPATH="$observation_root:$PYTHONPATH"
export MOST_FROZEN_AGENT="host:ObservationRouteAgent"
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    exec "$MOST_EVAL_PYTHON" -Bu "$@"
fi
