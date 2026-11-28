"""Real RGB same-action contract, permutation and no-evidence checks."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from placement_prior import CONTEXTS, PlacementPrior


def main(project):
    sys.path.insert(0,str(project/"jepa_env_v13/models"))
    from env_routes import EnvironmentRoutes
    local=project/"prior_approach_20261128"
    prior=PlacementPrior.load(local/"learned_prior.json")
    model=EnvironmentRoutes(project,11)
    torch.set_num_threads(2)
    origins=sorted((project/"jepa_env_v13/simulator/trees/fresh").glob("*/*/node_r.npz"))
    successes=[]
    for file in origins[::8]:
        with np.load(file) as f:
            rgb=f["ctx"];plans=np.array(f["plans"])+1
        cases=[plans[0,:1],plans[1,:5],plans[2,:13],plans[1,:24]]
        goal="clock"
        out=prior.forecast(model,rgb,goal,cases)
        perm=prior.forecast(model,rgb,goal,cases[::-1])[::-1]
        pure=prior.forecast(model,rgb,goal,plans,use_context=False)
        old=model.live(rgb,"clock.n.01",plans-1)
        complete=prior.forecast_with_evidence(model,rgb,goal,plans)
        rare=prior.forecast_with_evidence(model,rgb,"cup",cases)
        assert complete["evidence_supported"] and not rare["evidence_supported"]
        assert np.array_equal(rare["prior"],rare["evidence"])
        assert np.allclose(complete["evidence"],old["jepa"],atol=1e-6)
        ref=prior.calibrated(goal,old["env_pred_jepa"],np.zeros((len(plans),len(CONTEXTS))))
        current=prior.score_current(model,rgb[[0]],goal)
        assert out.shape==(4,) and np.all((out>=0)&(out<=1))
        assert np.allclose(out,perm,atol=1e-6)
        assert np.allclose(pure,ref,atol=1e-6) and current.shape==(1,)
        successes.append(dict(house=file.parent.parent.name,permutation_delta=float(np.abs(out-perm).max()),
                              full_length_delta=float(np.abs(pure-ref).max()),
                              full_target_head_delta=float(np.abs(complete["evidence"]-old["jepa"]).max())))
    assert len(set(r["house"] for r in successes))==12
    examples={}
    for target in ("cup","clock","painting"):
        key=prior.key(target)
        if key is None:
            examples[target]={"training_support":0}
            continue
        item=prior.targets[key]
        localp=np.array(item["local"])
        examples[target]=dict(training_rooms=item["rooms_present"],
                              top_room=prior.rooms[int(np.argmax(item["room"]))],
                              conditioned_on_living_room={c:round(float(p),4) for c,p in zip(CONTEXTS,localp[prior.rooms.index("living room")])})
    result=dict(verified=successes,examples=examples,real_prefix_lengths=[1,5,13,24],
                cost="paid requests zero",note="pose, instance inventory, target position or maps not supplied")
    (local/"interface_validation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main(Path(sys.argv[1]))
