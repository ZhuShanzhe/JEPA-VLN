"""Pair the current semantic prior with earlier frozen full route heads on identical queries."""
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from placement_prior import PlacementPrior


def main(project):
    sys.path.insert(0,str(project/"jepa_env_v13/models"))
    from env_routes import EnvironmentRoutes
    torch.set_num_threads(2)
    out=project/"prior_approach_20261128"
    model=EnvironmentRoutes(project,11)
    prior=PlacementPrior.load(out/"learned_prior.json")
    root=project/"jepa_route_v12/models/cache/hssd/fresh2"
    by_house={}; counts=Counter(); joint=Counter()
    for path in sorted((root/"actor").glob("*.npz")):
        with np.load(path) as a:
            ctx,plans,words,texts,house=a["ctx"],a["actions"],a["words"],a["text"],str(a["house"])
        with np.load(root/"truth"/path.name) as a:
            found=a["y"].T>0
        cues=prior.local_directions(plans,prior.current_cues(ctx,prior.visual_prompts(model)))
        for i,target in enumerate(words):
            if prior.key(str(target)) is None:
                continue
            data=model.cached(ctx,texts[i],plans,target)
            scores={name:data[name] for name in ("v11","jepa","direct","semantic_jepa","semantic_direct")}
            for arm in ("jepa","direct"):
                scores["updated_prior_"+arm]=prior.calibrated(target,data["env_pred_"+arm],cues,arm=arm)
            hits={k:bool(found[i,np.argmax(v)]) for k,v in scores.items()}
            counts.update(hits); counts["queries"]+=1
            local=by_house.setdefault(house,Counter()); local.update(hits);local["queries"]+=1
            joint[str((hits["updated_prior_jepa"],hits["updated_prior_direct"]))]+=1
    assert len(by_house)==12 and counts["queries"]==430
    (out/"paired_heads.json").write_text(json.dumps(dict(totals=counts,jepa_direct_joint=joint,by_house=by_house),indent=2))
    print(json.dumps(dict(totals=counts,jepa_direct_joint=joint)))


if __name__=="__main__":
    main(Path(sys.argv[1]))
