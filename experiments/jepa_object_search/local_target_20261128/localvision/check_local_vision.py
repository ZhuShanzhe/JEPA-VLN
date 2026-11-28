"""Real training-house RGB and negative controls for the local selector."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from local_vision import LocalViewPredictor


def main(project):
    sys.path[:0] = [str(project/"prior_approach_20261128"), str(project/"jepa_env_v13/models")]
    from env_routes import EnvironmentRoutes
    from placement_prior import PlacementPrior

    model = EnvironmentRoutes(project)
    prior = PlacementPrior.load(project/"prior_approach_20261128/learned_prior.json")
    selector = LocalViewPredictor(model, prior)
    train = project/"jepa_env_v13/simulator/trees/train"
    checks = []
    for house in sorted(d for d in train.iterdir() if d.is_dir())[:6]:
        node = sorted(house.glob("*/node_r.npz"))[0]
        with np.load(node, allow_pickle=False) as z:
            views = z["ctx"]
            plans = np.asarray(z["plans"],dtype=int)+1
        assert len(plans) >= 2 and np.isin(plans, (1,2,3)).all()
        target = "clock"
        orig = selector.score(views,target,plans)
        patch = orig["patches"]
        mat = np.asarray(orig["route_patch_scores"])
        assert mat.shape == (len(plans),9) and np.isfinite(mat).all() and np.all((mat>=0)&(mat<=1))
        assert [x["patch_id"] for x in patch]==list(range(9)) and all(x["view_index"]==1 for x in patch)
        assert selector.patches(views,target) == patch
        reverse = np.asarray(selector.score(views,target,plans[::-1])["route_patch_scores"])[::-1]
        assert np.allclose(mat,reverse,atol=2e-6)
        other = selector.score(views, "cup", plans)
        othermat=np.asarray(other["route_patch_scores"])
        flat = np.full_like(views, 127)
        blank = selector.score(flat, target, plans)
        assert max(p["support"] for p in blank["patches"])==0 and np.max(blank["route_patch_scores"])==0
        unknown = selector.score(views, "hydraulic jack", plans)
        assert prior.key("hydraulic jack") is None
        extra = max(pa["support"] for pa in unknown["patches"])
        assert extra==0 and np.max(unknown["route_patch_scores"])==0
        checks.append(dict(house=house.name, source=str(node.relative_to(train)), goals=[target,"cup"],
                           surfaces=sorted(set(p["surface"] for p in patch)),
                           spatial_range=float(np.ptp(mat,axis=1).max()),
                           same_patch_across_route_range=float(np.ptp(mat,axis=0).max()),
                           plan_permutation_error=float(np.max(abs(mat-reverse))),
                           known_goal_score_difference=float(abs(mat-othermat).max()),
                           unknown_goal_specific_contribution=extra,
                           uniform_input_max_score=float(np.max(blank["route_patch_scores"]))))
    assert all(max(r[k] for r in checks)>1e-4 for k in
               ("spatial_range","same_patch_across_route_range","known_goal_score_difference"))
    summary=dict(split="HSSD training houses",policy_inputs="RGB scan + target + proposed actions",
                 checkpoint=model.config["jepa"]["hashes"],cases=checks)
    out=Path(__file__).with_name("visual_contract_training_v2.json")
    out.write_text(json.dumps(summary,indent=2))
    print(json.dumps(checks))


if __name__=="__main__":
    torch.set_num_threads(2)
    main(Path(sys.argv[1]))
