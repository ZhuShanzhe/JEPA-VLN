"""Audit the effect of the rejection hint independently of navigating or asking for another image."""
from collections import Counter
import json
import numpy as np
import torch
from train_repair import HERE,groups
from replan import RepairScorer
from local_repair import alternatives
from path_geometry import nominal


def main():
    torch.set_num_threads(2)
    trained=RepairScorer(HERE/"private_repair"/"with_hint.pt")
    masked=RepairScorer(HERE/"private_repair"/"with_hint.pt")
    masked.hints=False
    without=RepairScorer(HERE/"private_repair"/"without_hint.pt")
    upstream={}
    for start in ("old_jepa","straight"):
        res=Counter()
        for _,_,s,reach,visible,_ in groups("test"):
            pr=int(np.argmax(s.old_joint)) if start=="old_jepa" else 0
            legal=alternatives(s.action[0,:s.length[0]])
            assert [list(p) for p in legal]==[a[:n].tolist() for a,n in zip(s.action,s.length)]
            if s.decision[pr]=="accept":continue
            candidates=[i for i in range(len(reach)) if i!=pr]
            chosen={}
            for method,p in (("feedback",trained.score(s,pr)),("masked",masked.score(s,pr)),
                             ("without",without.score(s,pr)),("product",s.reach*s.visible_if_reached)):
                i=candidates[np.argmax(p[candidates])]
                chosen[method]=i
                res[method+"_first_alternative_reaches"]+=int(reach[i])
                res[method+"_first_alternative_cleared_and_reaches"]+=int(s.decision[i]=="accept" and reach[i])
            res["rejected_requests"]+=1
            res["alternative_physically_exists"]+=int(reach[candidates].any())
            res["evidence_of_obstacle"]+=int(s.decision[pr]=="replan")
            for key in ("masked","without","product"):
                i=chosen["feedback"];j=chosen[key]
                res["different_than_"+key]+=int(i!=j)
                res["first_alternative_repaired_vs_"+key]+=int(reach[i] and not reach[j])
                res["first_alternative_spoiled_vs_"+key]+=int(not reach[i] and reach[j])
        upstream[start]=dict(res)
    # Curved plans are supported by construction. Only the straight-seed templates have physical labels in this panel.
    for p in ((1,), (1,2,1,1,3), (1,3,1,2,1,2),(2,3),(2,2,1,1,1)):
        for t in range(len(p)):
            for alt in alternatives(p,t):
                assert np.allclose(nominal(alt)[-1,:4],nominal(p)[-1,:4],atol=1e-5)
    (HERE/"repair_intervention_audit.json").write_text(json.dumps(upstream,indent=2)+"\n")
    return upstream

if __name__=="__main__":print(main())
