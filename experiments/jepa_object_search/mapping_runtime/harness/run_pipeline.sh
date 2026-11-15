#!/usr/bin/env bash
set -euo pipefail
umask 077
cd /root/autodl-tmp/jepa_object_search
source ops/hm3d_val_env.sh
rt="$NAV_ROOT/mapping_runtime"
repo="$NAV_ROOT/private_sources/MOST-Nav"
export VGGT_SLAM_HOST=127.0.0.1 VGGT_SLAM_PORT="${VGGT_SLAM_PORT:-18555}"
export NAV_DEBUG_ROOT="$rt/runs"
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mode="${1:?mode: server, map, native, native-many-check, frontier, execution-check, or frozen-episode}"
export NAV_RUN_ID="${2:?unique run ID required}"
[[ $NAV_RUN_ID =~ ^[A-Za-z0-9_.-]+$ && $NAV_RUN_ID != .* ]] || exit 2
run="$NAV_DEBUG_ROOT/$NAV_RUN_ID"
mkdir "$run"
cd "$repo"
export MOST_EXPERIMENT_MODEL="${MOST_EXPERIMENT_MODEL:-gpt-4o-2024-11-20}"
source "$NAV_ROOT/private_runtime/most_vlm/activate.sh"
export NAV_CAPTION_API_MODEL="$NAV_VLM_MODEL" NAV_CAPTION_API_URL="$NAV_VLM_API_URL"
export NAV_CAPTION_API_KEY="$NAV_VLM_API_KEY" NAV_CAPTION_HEALTH_TIMEOUT=120
export NAV_REQUIRE_CAPTION_PREFLIGHT=1
export PYTHONPATH="$rt/probes:$rt/harness:$repo:$BENCH_DIR/evaluation/main:$BENCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
case "$mode" in
  server)
    source "$rt/weights.env"
    export NAV_POINTING_MODEL_PATH="${NAV_POINTING_MODEL_PATH:-/root/autodl-tmp/jepa_instance_bench_20261016/models/Qwen2.5-VL-7B-Instruct}"
    if [[ -f "$rt/env_build/activate_mapping.sh" ]]; then source "$rt/env_build/activate_mapping.sh"; fi
    exec "$rt/env/bin/python" -B -u "$rt/harness/start_mapping.py" --host "$VGGT_SLAM_HOST" --port "$VGGT_SLAM_PORT" >"$run/server.log" 2>&1
    ;;
  map|native|native-many-check|frontier|execution-check|frozen-episode)
    if [[ "${MOST_BATCH_INHERIT_EVAL_FD:-}" == "1" ]]; then
      [[ /proc/$$/fd/9 -ef "$rt/eval.lock" ]] || exit 2
    else
      exec 9>"$rt/eval.lock"
    fi
    flock -n 9 || { echo "Another evaluator owns this mapping service"; exit 2; }
    export MOST_MAP_PROBE_RECEIPT="$run/geometry.json"
    agent=agents.nav_agent:NavAgent
    episode=$(/root/miniconda3/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1]))["engineering_episode"])' "$rt/acceptance/verified_task_selection.json")
    dataset="$MOST_DATASET_DIR"
    if [[ "$mode" == map ]]; then agent=map_probe:PhysicalMapProbe; fi
    if [[ "$mode" == frontier ]]; then agent=most_frontier_probe:ProbeAgent; fi
    if [[ "$mode" == execution-check ]]; then agent=execute_frontier:BranchExecutionCheck; fi
    if [[ "$mode" == native-many-check ]]; then
      dataset="$rt/external/three_dmem/dev_many_annotations"
      episode=$(/root/miniconda3/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["source_outside_frozen_set"] and d["required_count"]==2; print(d["engineering_episode"])' "$rt/external/three_dmem/dev_input_receipt.json")
    fi
    if [[ "$mode" == frozen-episode ]]; then
      episode="${3:?required fixed episode ID}"
      /root/miniconda3/bin/python -c 'import json,sys; ids={json.loads(s)["episode_id"] for s in open(sys.argv[1])}; assert sys.argv[2] in ids' "$NAV_ROOT/frontier_scenes/acceptance/most_many_category_attribute_frozen.jsonl" "$episode"
      dataset="$rt/frozen_manyon"
      agent="${MOST_FROZEN_AGENT:-agents.nav_agent:NavAgent}"
    fi
    cd "$BENCH_DIR"
    "$MOST_EVAL_PYTHON" -B evaluation/main/run_eval.py --config "$CONFIG" --scene-root "$SCENE_ROOT" \
      --dataset-dir "$dataset" --agent "$agent" --goal-type description --action-protocol goat \
      --episode-id "$episode" --log-positions --output "$run/result.json" >"$run/evaluator.log" 2>&1
    ;;
  *) echo "Unrecognized mode" >&2; exit 2;;
esac
