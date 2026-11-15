"""Verify paired action/observation collection on an HSSD training scene with no learner."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

root=Path(os.environ["NAV_ROOT"])
sys.path.insert(0,str(root/"langnav"))
from verify import scene_sim, setpose

out=Path(sys.argv[1])
out.mkdir(mode=0o700,exist_ok=False)
train=root/"a_exploration/official/hssd_v025/train/content/102344022.json.gz"
with gzip.open(train,"rt") as f:
    episode=json.load(f)["episodes"][0]
plans=[["turn_left"]*2+["move_forward"]*3,["turn_right"]*2+["move_forward"]*3]
context=[]
ends=[]
paths=[]
with scene_sim("102344022") as sim:
    for n,actions in enumerate(plans):
        initial=setpose(sim,episode["start_position"],episode["start_rotation"])["rgb"][:,:,:3].copy()
        context.append(initial)
        future=[]
        for action in actions:
            obs=sim.step(action)
            future.append(obs["rgb"][:,:,:3].copy())
        ends.append(future[-1])
        paths.append(future)
assert np.array_equal(*context) and not np.array_equal(*ends)
assert np.isfinite(np.asarray(paths)).all()
np.savez_compressed(out/"context_and_plans.npz",rgb=context[0],actions=np.asarray(plans))
np.savez_compressed(out/"observations_after_execution.npz",rgb=np.asarray(paths))
with np.load(out/"context_and_plans.npz",allow_pickle=False) as a, np.load(out/"observations_after_execution.npz",allow_pickle=False) as b:
    assert set(a.files)=={"rgb","actions"} and set(b.files)=={"rgb"}
    assert a["actions"].tolist()==plans and np.array_equal(b["rgb"],np.asarray(paths))
summary={"status":"pass","scenario":"HSSD train","branches":len(plans),"actions_per_branch":len(plans[0]),
         "identical_start_observation":True,"different_final_observations":True,
         "separate_online_and_future_files":True,"training_steps":0}
(out/"receipt.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary))
