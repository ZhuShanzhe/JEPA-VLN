"""Frozen local inference, kept separate from the original MOST evaluation environment."""
from multiprocessing.connection import Connection
import os
from pathlib import Path
import sys
import numpy as np

project = Path(os.environ["NAV_ROOT"])
sys.path[:0] = [str(project / p) for p in ("jepa_env_v13/models", "prior_approach_20261128", "jepa_variable_v16/model")]
from env_routes import EnvironmentRoutes
from placement_prior import PlacementPrior
from variable_evidence import VariableEvidence


def serve(fd, goal):
    prior = PlacementPrior.load(project / "prior_approach_20261128/learned_prior.json")
    model = EnvironmentRoutes(project)
    evidence = VariableEvidence(project)
    with Connection(fd) as stream:
        while True:
            try:
                msg = stream.recv()
            except EOFError:
                return
            values = prior.forecast(model, msg["views"], goal, msg["plans"])
            observed = prior.score_current(model, msg["views"], goal)
            forecast = evidence.predict(model, msg["views"], goal, [np.asarray(p)-1 for p in msg["plans"]])
            stream.send(dict(priors=values.tolist(), current=observed.tolist(), evidence=list(forecast["evidence_rate"])))


if __name__ == "__main__":
    serve(int(sys.argv[1]), sys.argv[2])
