"""Frozen vision and route inference. This process receives no scene IDs or annotations."""
from multiprocessing.connection import Connection
from pathlib import Path
import sys
import numpy as np


def serve(project, fd):
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here/"localvision"), str(project/"jepa_env_v13/models"),
                    str(project/"jepa_variable_v16/model"), str(project/"approach_search_20261128/eval4/locked")]
    from env_routes import EnvironmentRoutes
    from placement_prior import PlacementPrior
    from variable_evidence import VariableEvidence
    from local_vision import LocalViewPredictor
    model = EnvironmentRoutes(project)
    prior = PlacementPrior.load(project/"approach_search_20261128/eval4/locked/learned_prior.json")
    variable = VariableEvidence(project)
    local = LocalViewPredictor(model, prior)
    with Connection(fd) as channel:
        while True:
            try:
                req = channel.recv()
            except EOFError:
                return
            if req["mode"] == "observe":
                views, goal, plans = np.asarray(req["views"]), req["goal"], req["plans"]
                current = prior.score_current(model, views, goal).tolist()
                score = prior.forecast(model, views, goal, plans).tolist() if plans else []
                evidence = variable.predict(model, views, goal, [np.asarray(a)-1 for a in plans])["evidence_rate"] if plans else []
                channel.send(dict(current=current, score=score, evidence=list(evidence),
                                  patches=local.patches(views, goal)))
            elif req["mode"] == "local_sites":
                channel.send(local.score(views, goal, req["plans"])["route_patch_scores"])
            else:
                raise ValueError(req["mode"])


if __name__ == "__main__":
    serve(Path(sys.argv[1]), int(sys.argv[2]))
