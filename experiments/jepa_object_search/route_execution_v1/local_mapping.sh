#!/usr/bin/env bash
# Dedicated geometry-only mapper for the unmodified MOST RGB geometry test.
set -euo pipefail
dir="$(cd "$(dirname "$0")" && pwd)"
project="$(dirname "$dir")"
source "$project/ops/hm3d_val_env.sh"
rt="$project/mapping_runtime"
source "$rt/weights.env"
source "$rt/env_build/activate_mapping.sh"
cd "$project/private_sources/MOST-Nav"
export PYTHONPATH="$rt/harness:$PWD"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$rt/env/bin/python" -Bu "$rt/harness/start_mapping.py" --host 127.0.0.1 --port 18674 --no-semantic
