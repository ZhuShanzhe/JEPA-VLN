"""Validate independent long same-instance cross-check. Report counts at a native camera."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

from pose_data_contract import FEATURES
HERE=Path(__file__).resolve().parent
TAPS=(12,24,52,80,108)


def audit():
    manifests={}
    summary={s:{h:Counter() for h in TAPS} for s in ("train","val","test")}
    house_counts=Counter()
    cue_counts=Counter()
    origin_counts=Counter()
    for path in sorted((HERE/"private_long_cue").glob("*.npz")):
        split=path.stem.split("_",1)[0]
        house_counts[split]+=1
        manifests[path.name]=sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as f:
            n=len(f["reach"])
            assert tuple(f["feature_names"])==FEATURES
            assert f["pose"].shape==(n,20) and f["context"].shape==(n,2048)
            horizon=f["horizon"];a=f["action"];reach=f["reach"]==1;visible=f["visible"]
            assert np.array_equal((a!=0).sum(1),horizon)
            assert np.array_equal(reach,(~f["had_collision"])&(f["end_error_m"]<.14))
            assert np.array_equal(f["future_valid"],visible>=0) and np.all((visible<0)==~reach)
            assert np.all((f["pixels"]>=369)==(visible==1))
            assert np.allclose(np.linalg.norm(f["future"][reach].astype(np.float32),axis=1),1,atol=.001)
            assert np.all(f["future"][~reach]==0)
            assert np.allclose(np.linalg.norm(f["context"][:,:1024].astype(np.float32),axis=1),1,atol=.001)
            assert np.allclose(np.linalg.norm(f["context"][:,1024:].astype(np.float32),axis=1),1,atol=.001)
            native=f["native_mask"]
            assert np.array_equal(native,np.isclose(f["height"],1.31)&(f["pitch_deg"]==0))
            group=f["group"];pid=f["plan_id"]
            origin_counts[split]+=len(set(f["origin"]))
            cue_counts[split]+=len(set(group[horizon==12]))
            for h,c in summary[split].items():
                q=horizon==h
                c["rows"]+=int(q.sum())
                c["reachable"]+=int((q&reach).sum())
                c["joint_positive"]+=int((q&(visible==1)).sum())
                c["native_rows"]+=int((q&native).sum())
                c["native_reachable"]+=int((q&native&reach).sum())
                c["native_joint_positive"]+=int((q&native&(visible==1)).sum())
                for g in np.unique(group[q]):
                    v=q&(group==g)
                    assert v.sum()==9 and len(set(pid[v]))==3 and sum(native[v])==3
                    for i in range(3):
                        w=v&(pid==i)
                        assert len(set(reach[w]))==1
                        if reach[w].all():
                            c["reachable_cue_plan_triples"]+=1
                            c["visibility_flips_with_optics"]+=int(len(set(visible[w]))>1)
                    c["cue_origin_groups"]+=1
                    c["groups_any_native_joint_option"]+=int(np.any(visible[v&native]==1))
                    c["groups_mixed_native_joint_options"]+=int(
                        np.any(visible[v&native]==1) and np.any(visible[v&native]!=1))
    assert house_counts=={"train":12,"val":6,"test":12}
    report=dict(houses=dict(house_counts),accepted_origins=dict(origin_counts),unique_source_cues=dict(cue_counts),
                splits=summary,sha256=manifests,
                description="First four archived root origins per building; up to two initially visible instances selected by decreasing source area (40 pixels to 31% of view). Cues with invalid centre depth are skipped. Three frozen 108-action siblings per start are physically re-executed on the short-pose simulator. All 108 controls are issued even after a blocked forward. Taps 12,24,52,80,108. Off-target/post-collision optic labels masked. 1.31m/0 is native; .85m/+20 and 1.65m/-20 are camera rig probes. These are real re-renderings of existing long path starts in existing house splits, not independent unseen long paths.",
                fitting_protocol="The initial frozen short-pose head receives no long-cue supervision in its extrapolation test. The long-cue specialist fits only the training partition, selects epochs on validation, and reserves the test partition for evaluation.")
    (HERE/"long_cue_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(houses=house_counts,origins=origin_counts,cues=cue_counts,
        test_native_by_horizon={h:[v[k] for k in ("native_rows","native_reachable","native_joint_positive")]
                                for h,v in summary["test"].items()})))


if __name__=="__main__":audit()
