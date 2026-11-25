#!/usr/bin/env bash
set -euo pipefail
kit="$(cd "$(dirname "$0")" && pwd)"
project="${NAV_ROOT:-$(dirname "$kit")}"
source "$project/ops/hm3d_val_env.sh"
export NAV_ROOT="$project" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$kit:$project/jepa_confirm_v18/agent:$project/jepa_exec_v14:$project/mapping_runtime/harness:$project/mapping_runtime/probes:$project/private_sources/MOST-Nav:${PYTHONPATH:-}"
cd "$project/private_sources/MOST-Nav"
"$MOST_EVAL_PYTHON" -B "$kit/verify_offline.py" "$kit/verification.json"
