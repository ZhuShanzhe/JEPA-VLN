"""Verify actor/teacher optics, legal controls, family isolation, and depth-based occlusion proxies."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from pose_data_contract import FEATURES

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def audit():
    old={r["scene"].split("_")[0] for r in json.loads(
        (ROOT/"local_target_20261128/eval_family/panel.json").read_text())}
    families={s:set() for s in ("train","val","test")}
    official={}
    splits_path=Path("/root/autodl-tmp/jepa_vln_data_20261014/raw/hssd-hab/scene_splits.yaml")
    for line in splits_path.read_text().splitlines():
        if line.endswith(":"):
            partition=line[:-1].strip()
        elif line.startswith("- "):
            official[line[2:].strip().strip("'\"")]=partition
    split_counts={s:Counter() for s in families}
    manifest={}
    for path in sorted((HERE/"private_pose").glob("*.npz")):
        manifest[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as f:
            assert tuple(f["feature_names"])==FEATURES
            house=f["house"][0]
            split=str(f["split"][0])
            assert np.all(f["house"]==house) and np.all(f["split"]==split)
            families[split].add(house.split("_")[0])
            c=split_counts[split]
            c["houses"]+=1
            c["houses_in_official_"+official[str(house)]]+=1
            reach=f["reach"].astype(bool)
            valid=f["visible"]>=0
            seen=f["visible"]==1
            groups=f["group"]
            plans=f["plan_id"]
            h,pitch=f["height"],f["pitch_deg"]
            safe=f["safe_mask"]
            in_fov,intercepted=f["in_fov"],f["intercepted"]
            assert np.all(intercepted<=in_fov) and np.all(f["future_valid"]==valid)
            assert not np.any(reach&~valid)
            assert np.array_equal(f["native_mask"],np.isclose(h,1.31)&(pitch==0))
            assert np.allclose(np.linalg.norm(f["future"][valid].astype(float),axis=1),1,atol=.001)
            assert np.all(f["future"][~valid]==0)
            assert np.all((f["pixels"]>=369)==seen)
            n=len(seen)
            c["rows"]+=n
            for name,mask in dict(reachable=reach,optically_valid=valid,cue_clear=seen,
                    reachable_and_clear=reach&seen,unreachable_endpoint_renderable=~reach&valid,
                    blocked_plan_with_visible_hypothetical_endpoint=~reach&seen,
                    no_previous_target_points_in_fov=valid&(in_fov==0),
                    at_least_four_points_in_fov_half_intercepted=valid&(in_fov>=4)&(2*intercepted>=in_fov),
                    clear_despite_interception=seen&(in_fov>=4)&(2*intercepted>=in_fov),
                    source_depth_safe=safe,source_depth_safe_but_unreachable=safe&~reach).items():
                c[name]+=int(mask.sum())
            context=f["context"]
            assert np.allclose(np.linalg.norm(context[:,:1024].astype(float),axis=1),1,atol=.001)
            for g in np.unique(groups):
                q=groups==g
                c["source_groups"]+=1
                assert len(np.unique(context[q],axis=0))==1 and q.sum()==81
                assert len(np.unique(plans[q]))==9
                any_clear=False
                for pid in range(9):
                    v=q&(plans==pid)
                    assert len(np.unique(reach[v]))==1 and v.sum()==9
                    if not reach[v].all():continue
                    any_clear=any_clear or seen[v].any()
                    c["reachable_base_plans"]+=1
                    c["base_plan_visibility_flips_with_optics"]+=int(seen[v].any() and not seen[v].all())
                    hh=v&(pitch==0)
                    pp=v&np.isclose(h,1.31)
                    c["height_only_flip_at_zero_pitch"]+=int(seen[hh].any() and not seen[hh].all())
                    c["pitch_only_flip_at_native_height"]+=int(seen[pp].any() and not seen[pp].all())
                c["groups_with_reachable_clear_option"]+=int(any_clear)
    assert len(manifest)==30
    assert not ((families["train"]&families["test"])|(families["train"]&families["val"])|(families["val"]&families["test"]))
    assert all(not(v&old) for v in families.values())
    results=dict(splits={s:dict(c) for s,c in split_counts.items()},source_camera_m=1.31,
                 optics_heights_m=[.85,1.31,1.65],optics_pitch_degrees=[-20.,0.,20.],
                 horizontal_FOV_deg=90,target_min_pixels=369,source_box="ideal visible instance bbox",
                 split_note="12/6/12 HSSD project cohorts; official assignments cross-checked against scene_splits.yaml. Disjoint by scene ID prefix and from archived 8-house diagnostics.",
                 label_note="reach is collision-free traversal of this requested plan, plus <=0.14m endpoint error. It does not prove no other path exists. Visibility is rendered for navigable endpoints; off-mesh views are masked. source-derived point interception labels measure depth from a different surface at least 15cm nearer, not full-object occlusion.",
                 npz_sha256=manifest)
    path=HERE/"pose_data_audit.json"
    path.write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results["splits"]))

if __name__=="__main__": audit()
