"""Check every paired route, actor/label boundary, house split and physical endpoint agreement."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from path_geometry import NAMES,nominal

HERE=Path(__file__).resolve().parent


def examine():
    splits={s:Counter() for s in ("train","val","test")}
    families={s:set() for s in splits}
    compare={}
    by_length={}
    manifest={}
    for path in sorted((HERE/"private_data").glob("*.npz")):
        s,house=path.stem.split("_",1)
        manifest[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        families[s].add(house.split("_")[0])
        c=splits[s]; c["houses"]+=1
        with np.load(path,allow_pickle=False) as z:
            assert (z["split"]==s).all() and (z["house"]==house).all() and tuple(z["path_names"])==NAMES
            n=len(z["reach"])
            g,action,y,vis,pid=z["group"],z["action"],z["reach"],z["visible"],z["plan_id"]
            blocked=z["first_blocked"]
            assert np.array_equal((vis>=0),y==1) and not np.any((y==1)&(blocked>0))
            assert np.all((z["end_error_m"]<.14)|(y==0))
            assert np.array_equal(z["future_valid"],vis>=0)
            assert np.isfinite(z["path"]).all() and np.isfinite(z["context"]).all() and np.isfinite(z["pose"]).all()
            assert np.allclose(np.linalg.norm(z["future"][y==1].astype(float),axis=1),1.,atol=1e-3)
            forward=action==1
            pathf=z["path"].astype(np.float32)
            coverage=(pathf[:,:,5]*forward).sum(1)/np.maximum(1,forward.sum(1))
            body=(pathf[:,:,13].astype(bool)&forward).any(1)
            uncertainty=coverage<.65
            ids=np.flatnonzero(blocked)
            assert (action[ids,blocked[ids]-1]==1).all()
            c.update(rows=n,reachable=int(y.sum()),joint=int((vis==1).sum()),blocked=int((blocked>0).sum()),
                     source_cues=len(np.unique(z["source_index"])),
                     low_coverage_reachable=int((uncertainty&(y==1)).sum()),
                     low_coverage_unreachable=int((uncertainty&(y==0)).sum()),
                     observed_footprint_hit_and_blocked=int((body&(y==0)).sum()),
                     observed_footprint_hit_but_passed=int((body&(y==1)).sum()),
                     no_footprint_hit_but_blocked=int((~body&(y==0)).sum()))
            ends=np.column_stack([z[k] for k in ("end_x","end_y","end_z")])
            pose=z["pose"]
            features=z["context"]
            for group in np.unique(g):
                ii=np.flatnonzero(g==group)
                assert np.array_equal(pid[ii],np.arange(7))
                assert np.allclose(pose[ii,:8],pose[ii[0],:8],atol=1e-5)
                assert np.array_equal(features[ii],np.broadcast_to(features[ii[0]],features[ii].shape))
                plans=[a[a>0] for a in action[ii]]
                for q,p in zip(ii,plans):
                    assert np.allclose(nominal(tuple(map(int,p))),pathf[q,:len(p),:5],atol=.002)
                nsteps=len(plans[0]); comparable=by_length.setdefault((s,nsteps),Counter())
                wi=ii[y[ii]==1]; joint=vis[ii]==1
                comparable.update(groups=1,direct_reachable=int(y[ii[0]]),path_reachable=int(len(wi)>0),
                                  direct_joint=int(joint[0]),any_joint=int(joint.any()))
                c.update(groups=1,all_blocked=int(len(wi)==0),all_reachable=int(len(wi)==7),
                         mixed_reach=int(0<len(wi)<7),
                         direct_failed_but_detour_passed=int(not y[ii[0]] and len(wi)>0),
                         direct_failed_but_detour_clear=int(not y[ii[0]] and joint.any()),
                         several_reached=int(len(wi)>1),visible_label_disagrees_among_reached=int(len(set(vis[wi]))>1))
                if len(wi)>1:
                    delta=ends[wi]-ends[wi,None]
                    compare[s]=max(compare.get(s,0.),float(np.linalg.norm(delta[:,:,[0,2]],axis=2).max()))
    for a,b in (("train","val"),("val","test"),("train","test")):assert not families[a]&families[b]
    assert [splits[s]["houses"] for s in ("train","val","test")]==[12,6,12]
    report=dict(splits=splits,counts_by_forward_distance={f"{s}_{nsteps}_forward_steps":d for (s,nsteps),d in sorted(by_length.items())},
                source_images="Native fixed horizontal camera, depth and observed target bbox, same source image for all candidate plans.",
                first_blocked_definition="First 1-based forward primitive causing a simulator contact or actual horizontal displacement <0.23m. Collision ends attempted trajectory. No future pose, semantic ID or true pathway enters any actor feature.",
                split="Disjoint building families across train, validation and test on the original 12/6/12 HSSD experiment cohort. New start and candidate generation. This is a derived test, not an official benchmark.",
                geometry="All 7 nominal end positions and headings match within each target group. Reach also requires horizontal endpoint error <0.14m.",
                max_actual_endpoint_difference_m=compare,source_depth_strata="Low coverage is mean observed footprint samples <65% over proposed forward steps. Visible hit is an actual surface observed in an occupied cell on the nominal footprint. Neither diagnostic is perfect collision evidence.",
                sha256=manifest)
    (HERE/"path_data_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["splits"]))

if __name__=="__main__":examine()
