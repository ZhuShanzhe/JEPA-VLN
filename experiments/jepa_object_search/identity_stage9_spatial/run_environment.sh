#!/usr/bin/env bash
set -eu
spatial_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$spatial_dir")/identity_stage8_relook/run_environment.sh"
export PYTHONPATH="$spatial_dir:$PYTHONPATH"
export MOST_FROZEN_AGENT="spatial_agent:SpatialAgent"
cd "$spatial_dir"
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    exec "$MOST_EVAL_PYTHON" -Bu "$@"
fi
