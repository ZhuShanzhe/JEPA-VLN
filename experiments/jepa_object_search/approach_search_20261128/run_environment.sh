#!/usr/bin/env bash
set -eu
search_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(dirname "$search_dir")"
source "$project/ops/hm3d_val_env.sh"
export NAV_ROOT="$project"
export PYTHONPATH="$search_dir:$project/route_execution_v1:$project/prior_approach_20261128:$project/jepa_env_v13/most_integration:$project/mapping_runtime/probes:$project/mapping_runtime/harness:$project/private_sources/MOST-Nav:$BENCH_DIR/evaluation/main:$BENCH_DIR:${PYTHONPATH:-}"
export MOST_FROZEN_AGENT="search_agent:PriorApproachAgent"
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    exec "$MOST_EVAL_PYTHON" -Bu "$@"
fi
