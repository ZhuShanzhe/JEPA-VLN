"""Validate preregistered natural and blind panels without feeding any test labels back to training."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from path_geometry import NAMES,nominal

HERE=Path(__file__).resolve().parent


def audit():
    design=json.loads((HERE/"natural_design.json").read_text())
    cohorts=design["cohorts"]
    hashes={};counts={};blind=Counter()
    for split,homes in cohorts.items():
        c=Counter();counts[split]=c
        for h in homes:
            f=HERE/"private_natural"/(split+"_"+h+".npz")
            hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
            with np.load(f,allow_pickle=False) as z:
                group,seq,y,kind=z["group"],z["action"],z["reach"],z["kind"]
                geometry=z["path"]
                first=z["first_blocked"]
                assert np.all(z["house"]==h) and np.all(z["split"]==split) and tuple(z["path_names"])==NAMES
                assert set((seq>0).sum(1))==set(design["horizons"])
                assert not np.any((first>0)&y.astype(bool))
                for g in np.unique(group):
                    ids=np.flatnonzero(group==g)
                    assert len(ids)==4 and (kind[ids]=="natural").sum()==1
                    unique={}
                    for i in ids:
                        control=tuple(seq[i,seq[i]!=0].astype(int))
                        assert np.allclose(geometry[i,:len(control),:5],nominal(control),atol=.008)
                        if control in unique:assert unique[control]==y[i]
                        unique[control]=y[i]
                    c["unique_within_group"]+=len(unique)
                for k in ("natural","perturbed"):
                    v=kind==k
                    c[k+"_rows"]+=int(v.sum())
                    c[k+"_reachable"]+=int(y[v].sum())
                c["starts"]+=len(np.unique(z["source_index"]))
                c["houses"]+=1
                c["rows"]+=len(group)
    for h in cohorts["blind"]:
        p=HERE/"private_blind_paired"/("blind_"+h+".npz")
        with np.load(p,allow_pickle=False) as z:
            groups,y,pid,vis=z["group"],z["reach"],z["plan_id"],z["visible"]
            assert tuple(z["path_names"])==NAMES
            assert np.all(z["house"]==h) and np.all(z["split"]=="blind")
            blind.update(houses=1,rows=len(groups),reachable=int(y.sum()),joint=int((vis==1).sum()))
            for g in np.unique(groups):
                i=np.flatnonzero(groups==g)
                assert (pid[i]==np.arange(7)).all()
                assert np.allclose(z["pose"][i,:8],z["pose"][i[0],:8],atol=.00001)
                blind.update(groups=1,all_blocked=int(not y[i].any()),
                    direct_failed_detour_reached=int(y[i[0]]==0 and y[i].any()),
                    direct_failed_detour_joint=int(y[i[0]]==0 and (vis[i]==1).any()),
                    several_reached=int(y[i].sum()>1),
                    visible_disagreement_when_reached=int(len(set(vis[i][y[i]==1]))>1))
        hashes["paired/"+p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    assert not ({h.split("_")[0] for h in cohorts["blind"]}&
                {h.split("_")[0] for s in ("train","val","test") for h in cohorts[s]})
    report={"design_sha256":hashlib.sha256((HERE/"natural_design.json").read_bytes()).hexdigest(),
            "natural":counts,"blind_paired":blind,"file_sha256":hashes,
            "note":"The natural route is generated with an offline geometric follower and therefore has favourable reachability. Perturbations do not consult contacts. Rows sharing starts and routes are dependent."}
    (HERE/"natural_data_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("natural","blind_paired")}))

if __name__=="__main__":audit()
