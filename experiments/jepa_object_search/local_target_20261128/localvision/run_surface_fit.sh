#!/usr/bin/env bash
set -eu
project=/root/autodl-tmp/jepa_object_search
cd "$project/local_target_20261128/localvision"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
policy=/root/autodl-tmp/jepa_instance_bench_20261016/policy_env/bin/python
houses=$("$policy" -c 'import json; print(*json.load(open("../../jepa_env_v13/simulator/training_cohort.json"))["train"])')
for house in $houses
do
    if [ ! -f "training_private/$house.npz" ]; then
        "$project/simulation/habitat-env/bin/python" -Bu collect_surfaces.py "$project" "$house"
    fi
done
/root/autodl-tmp/jepa_instance_bench_20261016/policy_env/bin/python -Bu fit_surfaces.py "$project"
