"""Frozen learned, action-conditioned scoring on a training split, with counterfactual crops."""
from pathlib import Path
import sys
import json
import numpy as np
from multi_worker import start_worker


def main():
    here=Path(__file__).resolve().parent
    root=here.parent
    sample=next(iter(sorted((root/"jepa_env_v13/simulator/trees/train").glob("*/*/node_r.npz"))))
    with np.load(sample,allow_pickle=False) as z:
        views=z["ctx"]
        plans=(np.asarray(z["plans"][:4],dtype=int)+1).tolist()
    c,p=start_worker(root,here/"training_score_private.log",cpu=True)
    try:
        def ask(**kw):
            c.send(kw)
            if not c.poll(240):
                raise TimeoutError("training scorer")
            return c.recv()
        ask(mode="observe",views=views,goal="clock",plans=plans)
        box=[.2,.2,.5,.6]
        def score(i,b=box,actions=plans):
            return np.array(ask(mode="selected_box",view_index=i,bbox_norm=b,plans=actions))
        by_view=np.stack([score(i) for i in range(3)])
        by_box=score(0,[.6,.2,.95,.6])
        permutation_error=np.max(np.abs(score(0,actions=plans[::-1])[::-1]-by_view[0]))
        assert permutation_error<2e-6 and np.min(by_view)>=0 and np.max(by_view)<=1
        deltas=[np.ptp(by_view,axis=1).max(),np.ptp(by_view,axis=0).max(),np.max(abs(by_box-by_view[0]))]
        assert min(deltas)>1.e-5
        result=dict(training_split_only=True,source_camera_changes_similarity=float(deltas[1]),
                    moving_box_changes_similarity=float(deltas[2]),action_changes_similarity=float(deltas[0]),
                    plan_permutation_max_error=float(permutation_error))
        (here/"training_score_check.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result))
    finally:
        c.close()
        p.wait(timeout=45)


if __name__=="__main__":
    main()
