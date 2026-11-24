#!/usr/bin/env bash
set -euo pipefail
umask 077
P="${NAV_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
D=$P/jepa_variable_v16
source "$P/ops/hm3d_val_env.sh"
export PYTHONPATH="$D/adapter:$P/jepa_evidence_v15/adapter:$P/jepa_exec_v14:$P/jepa_env_v13/most_integration:$P/mapping_runtime/probes:$P/mapping_runtime/harness:$PYTHONPATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$P/private_sources/MOST-Nav"
runfolder="$(mktemp -d "$D/checks/private.XXXXXXXX")"
printf '%s\n' "$runfolder"
(cd "$D" && sha256sum adapter/*.py model/variable_evidence.py model/calibration.json checks/*.py checks/run_replay.sh) > "$runfolder/sources.sha256"
"$MOST_EVAL_PYTHON" -B "$D/checks/check_safety.py" "$P" "$runfolder/safety" >"$runfolder/safety.log" 2>&1 &
safety_pid=$!
"$MOST_EVAL_PYTHON" -B "$D/checks/replay_junctions.py" "$P" "$runfolder/controls" --controls >"$runfolder/controls.log" 2>&1 &
control_pid=$!
"$MOST_EVAL_PYTHON" -B "$D/checks/replay_junctions.py" "$P" "$runfolder/junctions" >"$runfolder/junctions.log" 2>&1
wait "$safety_pid"
wait "$control_pid"
(cd "$D" && sha256sum --status -c "$runfolder/sources.sha256")
"$MOST_EVAL_PYTHON" -B "$D/checks/summarize.py" "$runfolder"
