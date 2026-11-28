#!/usr/bin/env bash
set -eu
committed_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$committed_root")/gpt_inspect_only_20261128/run_environment.sh"
export PYTHONPATH="$committed_root:$PYTHONPATH"
export MOST_FROZEN_AGENT="commit_agent:CommittedInspectAgent"
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    exec "$MOST_EVAL_PYTHON" -Bu "$@"
fi
