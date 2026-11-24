#!/usr/bin/env bash
set -euo pipefail
umask 077
kit="$(cd "$(dirname "$0")" && pwd)"
project="$(cd "$kit/../.." && pwd)"
cd "$project"
source ops/hm3d_val_env.sh
case "${1:?first2 or next3}" in
  first2) output="$project/jepa_evidence_v15/panel/phase1_decision.json" ;;
  next3) output="$project/jepa_evidence_v15/panel/phase2_results.json" ;;
  *) exit 2 ;;
esac
export PYTHONPATH="$project/jepa_exec_v14:$project/mapping_runtime/probes:$project/mapping_runtime/harness:$project/private_sources/MOST-Nav:$BENCH_DIR/evaluation/main:$BENCH_DIR"
cd "$project/private_sources/MOST-Nav"
"$MOST_EVAL_PYTHON" -B -c 'import runpy,sys; from verify_rank import deny_ip_sockets; deny_ip_sockets(); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")' "$kit/assess_panel.py" "$project" "$project/mapping_runtime/pilots/jepa_evidence_v15_$1" "$output"
