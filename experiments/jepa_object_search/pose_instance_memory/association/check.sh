#!/usr/bin/env bash
set -eu
kit="$(cd "$(dirname "$0")" && pwd)"
project="$(dirname "$(dirname "$kit")")"
source "$project/ops/hm3d_val_env.sh"
export NAV_ROOT="$project" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$kit:$project/jepa_reid_v20/adapter:$project/jepa_report_v19:$project/jepa_confirm_v18/agent:$project/jepa_exec_v14:$project/mapping_runtime/harness:$project/mapping_runtime/probes:$project/private_sources/MOST-Nav:${PYTHONPATH:-}"
cd "$project/private_sources/MOST-Nav"
"$MOST_EVAL_PYTHON" -B -m unittest -v test_views.GeometryChecks test_views.NativeChecks test_adapter.AdapterChecks
if [ "${1:-}" = "--weights" ]; then
    "$MOST_EVAL_PYTHON" -B "$kit/check_pose_worker.py" "$project"
fi
