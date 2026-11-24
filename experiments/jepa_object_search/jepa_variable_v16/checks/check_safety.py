"""Probe the actual public MOST dispatcher using real motion and simulated higher-level events."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

from verify_rank import deny_ip_sockets, make_scene

deny_ip_sockets()
from verify_act import run_case
from variable_agent import VariableFrontier
import jepa_exec_agent

project, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=False)
env, episode, dataset = make_scene(project)
results = []
try:
    with patch.object(jepa_exec_agent, "JepaExecutionFrontier", VariableFrontier):
        for case in ("follow", "report", "finish"):
            trial = out / case
            trial.mkdir()
            results.append(run_case(env, episode, dataset, case, trial))
finally:
    env.close()
with (out/"receipt.json").open("x") as fp:
    json.dump(dict(passed=True, paid_calls=0, cases=results), fp, indent=2, allow_nan=False)
