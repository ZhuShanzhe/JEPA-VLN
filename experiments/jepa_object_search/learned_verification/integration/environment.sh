#!/usr/bin/env bash
set -eu
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
release="$(dirname "$here")"
project="$(dirname "$release")"
source "$project/ops/hm3d_val_env.sh"
export NAV_ROOT="$project" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
prior="$project/jepa_reid_v24"
export PYTHONPATH="$here:$release/active:$release/candidates:$project/jepa_reid_v24_postcheck/review:$prior:$prior/view:$prior/evidence:$prior/sim:$project/jepa_reid_v23:$project/jepa_reid_v23/report:$project/jepa_reid_v23/pose:$project/jepa_reid_v22/trace:$project/jepa_reid_v21/association:$project/jepa_reid_v21/detector:$project/jepa_reid_v20/adapter:$project/jepa_report_v19:$project/jepa_confirm_v18/agent:$project/jepa_completion_v17/physical:$project/jepa_exec_v14:$project/mapping_runtime/harness:$project/mapping_runtime/probes:$project/private_sources/MOST-Nav:${PYTHONPATH:-}"
cd "$BENCH_DIR"
export PYTHONPATH="$release/active_weak:$PYTHONPATH"
