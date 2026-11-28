"""Isolated GPU model worker. Only receives RGB, proposed actions and user goal."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("project", type=Path)
    args = p.parse_args()
    output = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        sys.path[:0] = [str(Path(__file__).resolve().parent / "locked"),str(args.project / "jepa_env_v13/models"),
                        str(args.project / "jepa_variable_v16/model")]
        from env_routes import EnvironmentRoutes
        from placement_prior import PlacementPrior
        from variable_evidence import VariableEvidence
        model = EnvironmentRoutes(args.project)
        old = VariableEvidence(args.project)
        prior = PlacementPrior.load(Path(__file__).resolve().parent / "locked/learned_prior.json")
        print(json.dumps({"ready": True}), file=output, flush=True)
        for line in sys.stdin:
            try:
                req = json.loads(line)
                rgb = np.load(req["views"], allow_pickle=False)
                plans, goal = req["plans"], req["target"]
                current = prior.score_current(model,rgb,goal).tolist()
                evidence = old.predict(model,rgb,goal,[np.asarray(x)-1 for x in plans])["evidence_rate"]
                score = evidence if req["arm"]=="old" else prior.forecast(model,rgb,goal,plans).tolist()
                print(json.dumps(dict(current=current,score=score,evidence=evidence)),file=output,flush=True)
            except Exception:
                traceback.print_exc()
                print(json.dumps({"error":"inference_failed"}),file=output,flush=True)
                break

if __name__ == "__main__":
    main()
