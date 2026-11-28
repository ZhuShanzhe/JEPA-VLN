#!/usr/bin/env bash
set -eu
multiview_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$multiview_root")/committed_inspect_20261128/run_environment.sh"
export PYTHONPATH="$multiview_root:$PYTHONPATH"
export MOST_FROZEN_AGENT="multi_agent:MultiViewAgent"
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    exec "$MOST_EVAL_PYTHON" -Bu "$@"
fi
