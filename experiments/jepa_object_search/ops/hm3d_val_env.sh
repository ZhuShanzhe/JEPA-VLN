# Source in bash to use the licensed scenes with the original MOST-Bench entry point.
source /root/autodl-tmp/jepa_object_search/ops/env.sh
export HM3D_ROOT="$NAV_ROOT/frontier_scenes/hm3d_v02"
export SCENE_ROOT="$HM3D_ROOT/val"
export HM3D_SCENE_DATASET="$SCENE_ROOT/hm3d_annotated_val_basis.scene_dataset_config.json"
export BENCH_DIR="$NAV_ROOT/frontier_scenes/most_release"
export MOST_DATASET_DIR="$BENCH_DIR/benchmark_dataset/semantic"
export CONFIG="$BENCH_DIR/evaluation/main/hm3d_config.yaml"
export MOST_EVAL_PYTHON="$NAV_ROOT/frontier_scenes/evaluator_env/venv/bin/python"
export NAV_DEBUG_ROOT="${NAV_DEBUG_ROOT:-$NAV_ROOT/logs/most_nav}"
export PYTHONPATH="$NAV_ROOT/private_sources/MOST-Nav:$BENCH_DIR/evaluation/main:$BENCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
