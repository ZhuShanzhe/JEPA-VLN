#!/usr/bin/env bash
# Run the existing local fixtures and Habitat replay without a network-capable evaluator.
set -euo pipefail
umask 077
adapter="$(cd "$(dirname "$0")" && pwd)"
project="$(dirname "$adapter")"
cd "$project"
source ops/hm3d_val_env.sh
export PYTHONPATH="$adapter:$project/jepa_env_v13/most_integration:$project/mapping_runtime/probes:$project/mapping_runtime/harness:$project/private_sources/MOST-Nav:$BENCH_DIR/evaluation/main:$BENCH_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
output="$(mktemp -d "$adapter/offline.XXXXXXXX")"
echo "$output"
cd "$project/private_sources/MOST-Nav"
offline_python() {
    "$MOST_EVAL_PYTHON" -B -c 'import runpy, sys; from verify_rank import deny_ip_sockets; deny_ip_sockets(); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")' "$@"
}
offline_python "$adapter/test_execution.py" >"$output/unit.log" 2>&1
offline_python "$adapter/verify_rank.py" --project "$project" --output "$output/frozen_rank_receipt.json"
offline_python "$adapter/verify_act.py" --project "$project" --output "$output/act_receipt.json"
offline_python "$adapter/verify_habitat.py" "$project" "$output/physical"
