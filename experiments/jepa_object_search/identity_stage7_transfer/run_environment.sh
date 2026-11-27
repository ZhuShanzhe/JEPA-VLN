#!/usr/bin/env bash
set -eu
transfer_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$transfer_dir")/identity_stage6_approach/run_environment.sh"
export PYTHONPATH="$transfer_dir:$PYTHONPATH"
export MOST_FROZEN_AGENT="transfer_agent:ArrivalTransferAgent"
cd "$transfer_dir"
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    exec "$MOST_EVAL_PYTHON" -Bu "$@"
fi
