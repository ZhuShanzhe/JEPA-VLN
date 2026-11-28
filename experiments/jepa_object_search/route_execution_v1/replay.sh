#!/usr/bin/env bash
set -euo pipefail
dir="$(cd "$(dirname "$0")" && pwd)"
project="$(dirname "$dir")"
source "$project/ops/hm3d_val_env.sh"
export PYTHONPATH="$dir:$project/jepa_exec_v14:$project/mapping_runtime/probes:$project/mapping_runtime/harness:$project/private_sources/MOST-Nav:$BENCH_DIR/evaluation/main:$BENCH_DIR"
cd "$project/private_sources/MOST-Nav"
exec "$MOST_EVAL_PYTHON" -Bu "$dir/replay_views.py" "$project" "$1"
